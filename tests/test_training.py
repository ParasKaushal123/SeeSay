from dataclasses import replace
import json

import joblib
import numpy as np
import pytest

from cxr_evidence.cli import main
from cxr_evidence.data import load_dataset
from cxr_evidence.demo import make_demo
from cxr_evidence.training import TrainConfig, metrics, train, train_to_directory


@pytest.fixture
def archive(tmp_path):
    return make_demo(tmp_path / "features.npz")


def rewrite(source, destination, **changes):
    with np.load(source, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    arrays.update(changes)
    np.savez_compressed(destination, **arrays)
    return destination


def test_patient_leakage_is_rejected(archive, tmp_path):
    data = load_dataset(archive)
    patient_ids = data.patient_ids.copy()
    patient_ids[-1] = patient_ids[0]
    bad = rewrite(archive, tmp_path / "bad.npz", patient_ids=patient_ids)
    with pytest.raises(ValueError, match="multiple splits"):
        load_dataset(bad)


def test_unknowns_are_not_scored_as_negative():
    result = metrics(np.array([[1], [-1], [0]]), np.array([[0.8], [0.99], [0.2]]), ("finding",))
    assert result["pooled"]["known_labels"] == 2
    assert result["pooled"]["brier"] == pytest.approx(0.04)


def test_training_and_saved_model_round_trip(archive, tmp_path):
    output = tmp_path / "run"
    report = train_to_directory(archive, output, TrainConfig(trees=20))
    model = joblib.load(output / "model.joblib")
    data = load_dataset(archive)
    probabilities = model.predict(data)
    assert probabilities.shape == data.labels.shape
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert report["label_source"] == "synthetic"
    assert "calibration" not in report["evaluation"]
    assert report["evaluation"]["test"]["calibrated"]["pooled"]["known_labels"] == int((data.labels[data.splits == "test"] != -1).sum())
    assert json.loads((output / "metrics.json").read_text())["dataset_sha256"]
    with pytest.raises(FileExistsError):
        train_to_directory(archive, output)


def test_test_labels_do_not_change_model(archive):
    data = load_dataset(archive)
    changed = data.labels.copy()
    test = data.splits == "test"
    changed[test] = np.where(changed[test] == -1, -1, 1 - changed[test])
    config = TrainConfig(trees=20)
    original, _ = train(data, config)
    modified, _ = train(replace(data, labels=changed), config)
    np.testing.assert_array_equal(original.predict(data), modified.predict(data))


def test_feature_order_and_encoder_mismatch_rejected(archive):
    data = load_dataset(archive)
    model, _ = train(data, TrainConfig(trees=10))
    with pytest.raises(ValueError, match="names and order"):
        model.predict(replace(data, evidence_names=tuple(reversed(data.evidence_names))))
    with pytest.raises(ValueError, match="Encoder"):
        model.predict(replace(data, encoder_id="another-model-same-dimension"))


def test_missing_evidence_accepted_but_invalid_labels_rejected(archive, tmp_path):
    data = load_dataset(archive)
    missing = np.full_like(data.evidence, np.nan)
    valid = load_dataset(rewrite(archive, tmp_path / "missing.npz", evidence=missing))
    model, _ = train(valid, TrainConfig(trees=10))
    assert np.isfinite(model.predict(valid)).all()
    invalid_labels = data.labels.copy()
    invalid_labels[0, 0] = 2
    with pytest.raises(ValueError, match="only -1, 0, 1"):
        load_dataset(rewrite(archive, tmp_path / "invalid.npz", labels=invalid_labels))


def test_one_class_calibration_fails_explicitly(archive):
    data = load_dataset(archive)
    changed = data.labels.copy()
    changed[data.splits == "calibration"] = 0
    with pytest.raises(ValueError, match="calibration needs"):
        train(replace(data, labels=changed))


def test_cli_demo_and_label_free_prediction(tmp_path, monkeypatch, capsys):
    output = tmp_path / "demo"
    monkeypatch.setattr("sys.argv", ["cxr-evidence", "demo", "--output", str(output)])
    main()
    assert json.loads(capsys.readouterr().out)["label_source"] == "synthetic"
    with np.load(output / "synthetic_features.npz", allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files if key != "labels"}
    arrays["splits"] = np.full(len(arrays["study_ids"]), "inference")
    inference = tmp_path / "inference.npz"
    np.savez_compressed(inference, **arrays)
    prediction = tmp_path / "predictions.npz"
    monkeypatch.setattr("sys.argv", ["cxr-evidence", "predict", "--data", str(inference),
                                   "--model", str(output / "run/model.joblib"), "--output", str(prediction)])
    main()
    with np.load(prediction, allow_pickle=False) as predictions:
        assert predictions["probabilities"].shape == (240, 3)
        assert str(predictions["calibration_label_source"]) == "synthetic"
