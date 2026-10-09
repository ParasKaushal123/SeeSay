"""Patient-grouped probe training, out-of-fold stacking, and held-out calibration."""

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import joblib
from lightgbm import LGBMClassifier
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import FeatureDataset, load_dataset


@dataclass(frozen=True)
class TrainConfig:
    folds: int = 3
    seed: int = 42
    trees: int = 80
    threads: int = 2

    def validate(self) -> None:
        if self.folds < 2 or self.trees < 1 or self.threads < 1:
            raise ValueError("folds >= 2, trees >= 1, and threads >= 1 are required")


def fit_probes(features: np.ndarray, labels: np.ndarray, seed: int) -> list:
    probes = []
    for column in labels.T:
        known = column != -1
        target = column[known]
        if len(np.unique(target)) < 2:
            # Smoothed prior for folds without both classes; never an inferred negative.
            probes.append(float((target.sum() + 1) / (len(target) + 2)))
        else:
            probe = make_pipeline(
                StandardScaler(), LogisticRegression(C=0.1, max_iter=500, random_state=seed),
            )
            probe.fit(features[known], target)
            probes.append(probe)
    return probes


def probe_probabilities(probes: list, features: np.ndarray) -> np.ndarray:
    return np.column_stack([
        np.full(len(features), probe) if isinstance(probe, float)
        else probe.predict_proba(features)[:, 1]
        for probe in probes
    ])


def fusion_features(data: FeatureDataset, probabilities: np.ndarray) -> np.ndarray:
    n, classes = probabilities.shape
    return np.column_stack([
        probabilities.reshape(-1),
        data.evidence.reshape(n * classes, len(data.evidence_names)),
        np.isnan(data.evidence).reshape(n * classes, len(data.evidence_names)).astype(float),
        np.tile(np.arange(classes), n),
    ])


@dataclass
class EvidenceModel:
    probes: list
    stacker: LGBMClassifier
    calibrator: LogisticRegression
    finding_names: tuple[str, ...]
    evidence_names: tuple[str, ...]
    image_dimensions: int
    label_source: str
    encoder_id: str
    schema_version: int = 1

    def predict(self, data: FeatureDataset, *, calibrated: bool = True) -> np.ndarray:
        if data.finding_names != self.finding_names or data.evidence_names != self.evidence_names:
            raise ValueError("Finding/evidence names and order must match the trained model")
        if data.image_features.shape[1] != self.image_dimensions:
            raise ValueError("Image embedding dimensions do not match the trained model")
        if data.encoder_id != self.encoder_id:
            raise ValueError("Encoder revision/preprocessing does not match the trained model")
        probes = probe_probabilities(self.probes, data.image_features)
        matrix = fusion_features(data, probes)
        if calibrated:
            margins = self.stacker.booster_.predict(matrix, raw_score=True).reshape(-1, 1)
            probabilities = self.calibrator.predict_proba(margins)[:, 1]
        else:
            probabilities = self.stacker.booster_.predict(matrix)
        return probabilities.reshape(len(data.study_ids), len(self.finding_names))


def metrics(labels: np.ndarray, probabilities: np.ndarray, finding_names: tuple[str, ...]) -> dict:
    def calculate(target: np.ndarray, probability: np.ndarray) -> dict:
        known = target != -1
        target, probability = target[known], probability[known]
        result = {"known_labels": int(len(target)), "positives": int(target.sum())}
        if not len(target):
            return result
        result.update({
            "brier": float(brier_score_loss(target, probability)),
            "log_loss": float(log_loss(target, probability, labels=[0, 1])),
        })
        if len(np.unique(target)) == 2:
            result["auroc"] = float(roc_auc_score(target, probability))
            result["average_precision"] = float(average_precision_score(target, probability))
        return result

    return {
        "pooled": calculate(labels.reshape(-1), probabilities.reshape(-1)),
        "by_finding": {
            name: calculate(labels[:, i], probabilities[:, i])
            for i, name in enumerate(finding_names)
        },
    }


def train(data: FeatureDataset, config: TrainConfig = TrainConfig()) -> tuple[EvidenceModel, dict]:
    config.validate()
    if data.labels is None:
        raise ValueError("Training requires labels")
    if "inference" in data.splits:
        raise ValueError("Training archives must not contain an inference split")
    training = data.subset(data.splits == "train")
    calibration = data.subset(data.splits == "calibration")
    if not len(training.study_ids) or not len(calibration.study_ids):
        raise ValueError("Disjoint train and calibration splits are required")
    if len(np.unique(training.patient_ids)) < config.folds:
        raise ValueError("Not enough training patients for the requested folds")
    for name, part in [("train", training), ("calibration", calibration)]:
        known_labels = part.labels[part.labels != -1]
        if len(np.unique(known_labels)) < 2:
            raise ValueError(f"{name} needs known positive and negative labels")
    for i, name in enumerate(data.finding_names):
        if len(np.unique(training.labels[:, i][training.labels[:, i] != -1])) < 2:
            raise ValueError(f"Finding {name!r} needs both training classes; remove unsupported targets")

    started = time.perf_counter()
    out_of_fold = np.empty(training.labels.shape, dtype=float)
    groups = GroupKFold(n_splits=config.folds)
    for fit_indices, held_indices in groups.split(training.image_features, groups=training.patient_ids):
        probes = fit_probes(training.image_features[fit_indices], training.labels[fit_indices], config.seed)
        out_of_fold[held_indices] = probe_probabilities(probes, training.image_features[held_indices])
    matrix = fusion_features(training, out_of_fold)
    target = training.labels.reshape(-1)
    known = target != -1
    stacker = LGBMClassifier(
        objective="binary", n_estimators=config.trees, learning_rate=0.05,
        max_depth=3, num_leaves=7, min_child_samples=20, reg_lambda=5,
        random_state=config.seed, n_jobs=config.threads, verbosity=-1,
        deterministic=True, force_col_wise=True,
    )
    feature_names = ["feature__probe_probability", *data.evidence_names,
                     *(f"missing__{name}" for name in data.evidence_names), "finding_id"]
    stacker.fit(matrix[known], target[known],
                categorical_feature=[matrix.shape[1] - 1], feature_name=feature_names)
    final_probes = fit_probes(training.image_features, training.labels, config.seed)
    cal_matrix = fusion_features(calibration, probe_probabilities(final_probes, calibration.image_features))
    cal_target = calibration.labels.reshape(-1)
    cal_known = cal_target != -1
    margins = stacker.booster_.predict(cal_matrix, raw_score=True).reshape(-1, 1)
    calibrator = LogisticRegression(C=1.0, max_iter=500, random_state=config.seed)
    calibrator.fit(margins[cal_known], cal_target[cal_known])
    model = EvidenceModel(final_probes, stacker, calibrator, data.finding_names,
                          data.evidence_names, data.image_features.shape[1], data.label_source, data.encoder_id)
    report = {
        "schema_version": 1,
        "label_source": data.label_source,
        "encoder_id": data.encoder_id,
        "config": asdict(config),
        "training_seconds": time.perf_counter() - started,
        "stacking": "patient-grouped out-of-fold image probes; cached external evidence",
        "calibration": "pooled sigmoid on disjoint calibration patients; not per-finding calibration",
        "feature_names": feature_names,
        "findings": list(data.finding_names),
        "training_label_counts": {
            name: {"positive": int((training.labels[:, i] == 1).sum()),
                   "negative": int((training.labels[:, i] == 0).sum()),
                   "unknown": int((training.labels[:, i] == -1).sum())}
            for i, name in enumerate(data.finding_names)
        },
        "split_studies": {str(split): int((data.splits == split).sum()) for split in np.unique(data.splits)},
        "evaluation": {},
    }
    # Calibration-fit metrics would be optimistic; only untouched split metrics are reported.
    for split in ("validation", "test"):
        part = data.subset(data.splits == split)
        if len(part.study_ids):
            report["evaluation"][split] = {
                "uncalibrated": metrics(part.labels, model.predict(part, calibrated=False), data.finding_names),
                "calibrated": metrics(part.labels, model.predict(part), data.finding_names),
            }
    return model, report


def train_to_directory(data_path: str | Path, output: str | Path,
                       config: TrainConfig = TrainConfig()) -> dict:
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    data = load_dataset(data_path)
    model, report = train(data, config)
    with Path(data_path).open("rb") as handle:
        report["dataset_sha256"] = hashlib.file_digest(handle, "sha256").hexdigest()
    report["versions"] = {name: importlib.metadata.version(name)
                          for name in ("numpy", "scikit-learn", "lightgbm", "joblib")}
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output / "model.joblib", compress=3)
    (output / "metrics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report
