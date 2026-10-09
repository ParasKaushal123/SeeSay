"""Synthetic plumbing fixture. Its scores are not medical benchmark results."""

from pathlib import Path

import numpy as np


def make_demo(path: str | Path, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    patients, studies_per_patient, dimensions, classes = 120, 2, 12, 3
    n = patients * studies_per_patient
    features = rng.normal(size=(n, dimensions)).astype(np.float32)
    logits = features[:, :classes] * 1.8 + features[:, 3:6] * 0.5
    probability = 1 / (1 + np.exp(-logits))
    labels = (rng.random((n, classes)) < probability).astype(np.int8)
    labels[rng.random(labels.shape) < 0.12] = -1
    healthy = np.clip(probability + rng.normal(0, 0.3, (n, classes)), 0, 1)
    regional = np.clip(probability + rng.normal(0, 0.22, (n, classes)), 0, 1)
    reader = np.clip(probability + rng.normal(0, 0.2, (n, classes)), 0, 1)
    evidence = np.stack([
        healthy, rng.uniform(0.5, 1, (n, classes)), regional,
        reader, np.abs(reader - regional),
    ], axis=-1).astype(np.float32)
    evidence[rng.random((n, classes)) < 0.2, 2] = np.nan
    splits = np.repeat(["train"] * 80 + ["calibration"] * 20 + ["test"] * 20, studies_per_patient)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    np.savez_compressed(
        path, study_ids=np.array([f"study_{i:04d}" for i in range(n)]),
        patient_ids=np.repeat([f"patient_{i:04d}" for i in range(patients)], studies_per_patient),
        splits=splits, image_features=features, evidence=evidence, labels=labels,
        finding_names=np.array(["atelectasis", "cardiomegaly", "pleural_effusion"]),
        evidence_names=np.array([
            "healthy__distance", "healthy__match_quality", "regional__score",
            "reader__positive_score", "reader__disagreement",
        ]), label_source=np.array("synthetic"), encoder_id=np.array("synthetic-v1"),
    )
    return path
