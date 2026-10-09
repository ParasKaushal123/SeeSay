"""A strict, pickle-free interchange format for cached study features."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class FeatureDataset:
    study_ids: np.ndarray
    patient_ids: np.ndarray
    splits: np.ndarray
    image_features: np.ndarray
    evidence: np.ndarray
    finding_names: tuple[str, ...]
    evidence_names: tuple[str, ...]
    label_source: str
    encoder_id: str
    labels: np.ndarray | None = None

    def subset(self, mask: np.ndarray) -> "FeatureDataset":
        return FeatureDataset(
            self.study_ids[mask], self.patient_ids[mask], self.splits[mask],
            self.image_features[mask], self.evidence[mask], self.finding_names,
            self.evidence_names, self.label_source, self.encoder_id,
            None if self.labels is None else self.labels[mask],
        )


def _strings(value: np.ndarray, name: str) -> np.ndarray:
    if value.ndim != 1 or value.dtype.kind not in "US":
        raise ValueError(f"{name} must be a one-dimensional string array")
    value = value.astype(str)
    if any(not item.strip() for item in value):
        raise ValueError(f"{name} contains empty values")
    return value


def load_dataset(path: str | Path, *, require_labels: bool = True) -> FeatureDataset:
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "study_ids", "patient_ids", "splits", "image_features", "evidence",
            "finding_names", "evidence_names", "label_source", "encoder_id",
        }
        if require_labels:
            required.add("labels")
        missing = required - set(archive.files)
        if missing:
            raise ValueError(f"Missing arrays: {sorted(missing)}")
        raw = {key: archive[key] for key in required}
        if "labels" in archive.files:
            raw["labels"] = archive["labels"]

    study = _strings(raw["study_ids"], "study_ids")
    patient = _strings(raw["patient_ids"], "patient_ids")
    splits = _strings(raw["splits"], "splits")
    findings = tuple(_strings(raw["finding_names"], "finding_names"))
    names = tuple(_strings(raw["evidence_names"], "evidence_names"))
    n, c = len(study), len(findings)
    if not n or not c:
        raise ValueError("Dataset and finding list must be nonempty")
    if len(set(study)) != n:
        raise ValueError("study_ids must be unique; aggregate views into one study")
    if len(patient) != n or len(splits) != n:
        raise ValueError("ID and split arrays must have equal lengths")
    if len(set(findings)) != c or len(set(names)) != len(names):
        raise ValueError("Finding and evidence names must be unique")
    if any(not name.startswith(("healthy__", "regional__", "reader__")) for name in names):
        raise ValueError("Evidence names need healthy__, regional__, or reader__ prefixes")
    if not set(splits) <= {"train", "calibration", "validation", "test", "inference"}:
        raise ValueError("Unrecognized split")
    assignment: dict[str, str] = {}
    for pid, split in zip(patient, splits):
        if pid in assignment and assignment[pid] != split:
            raise ValueError(f"Patient {pid!r} occurs in multiple splits")
        assignment[pid] = split

    features = np.asarray(raw["image_features"], dtype=np.float32)
    evidence = np.asarray(raw["evidence"], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != n or features.shape[1] == 0:
        raise ValueError("image_features must have shape [studies, embedding_dimensions]")
    if not np.isfinite(features).all():
        raise ValueError("image_features must be finite")
    if evidence.shape != (n, c, len(names)) or np.isinf(evidence).any():
        raise ValueError("evidence must have shape [studies, findings, channels]; NaN is allowed, infinity is not")
    labels = raw.get("labels")
    if labels is not None:
        if labels.shape != (n, c) or not np.isin(labels, [-1, 0, 1]).all():
            raise ValueError("labels must have shape [studies, findings] and contain only -1, 0, 1")
        labels = labels.astype(np.int8)
    source_array = np.asarray(raw["label_source"])
    if source_array.shape != () or source_array.dtype.kind not in "US":
        raise ValueError("label_source must be a scalar string")
    source = str(source_array)
    if source not in {"synthetic", "report_derived", "image_adjudicated"}:
        raise ValueError("label_source must be synthetic, report_derived, or image_adjudicated")
    encoder_array = np.asarray(raw["encoder_id"])
    if encoder_array.shape != () or encoder_array.dtype.kind not in "US" or not str(encoder_array).strip():
        raise ValueError("encoder_id must identify the frozen encoder revision and preprocessing")
    return FeatureDataset(study, patient, splits, features, evidence, findings, names, source, str(encoder_array), labels)
