"""Tests for the cxr_report pipeline.

Tests that exercise generation use mocks so they run without
a GPU or downloaded weights.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from cxr_report.images import load_image, image_sha256
from cxr_report.prompts import chest_xray_prompt
from cxr_report.output import extract_sections, format_result, append_jsonl, save_result
from cxr_report.batch import load_manifest, load_completed, run_batch
from cxr_report.model import ModelInfo
from cxr_report.generator import GenerationResult


# ------------------------------------------------------------------ fixtures

@pytest.fixture
def tiny_image_path(tmp_path: Path) -> Path:
    """Create a minimal 10×10 red PNG for testing."""
    path = tmp_path / "test_img.png"
    Image.new("RGB", (10, 10), color="red").save(path)
    return path


@pytest.fixture
def mock_model_info() -> ModelInfo:
    return ModelInfo(
        model_id="google/medgemma-1.5-4b-it",
        revision="abc123",
        dtype="float16",
        device_map="auto",
        quantized=False,
    )


@pytest.fixture
def mock_generation_result() -> GenerationResult:
    return GenerationResult(
        raw_report="Findings: Normal heart size.\nImpression: No acute process.",
        findings="Normal heart size.",
        impression="No acute process.",
        status="success",
        warnings=(),
        elapsed_seconds=1.23,
        prompt_version="chest_xray_v1",
        generation_config={"max_new_tokens": 384, "do_sample": False},
    )


# --------------------------------------------------------------- images tests

def test_load_image_valid(tiny_image_path: Path) -> None:
    img = load_image(tiny_image_path)
    assert img.mode == "RGB"
    assert img.size == (10, 10)


def test_load_image_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "nonexistent.png")


def test_load_image_corrupt(tmp_path: Path) -> None:
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image file at all")
    with pytest.raises(ValueError, match="corrupt"):
        load_image(bad)


def test_image_sha256(tiny_image_path: Path) -> None:
    h1 = image_sha256(tiny_image_path)
    h2 = image_sha256(tiny_image_path)
    assert h1 == h2
    assert isinstance(h1, str) and len(h1) == 64


# -------------------------------------------------------------- prompts tests

def test_prompt_without_indication() -> None:
    cfg = chest_xray_prompt()
    assert "Findings" in cfg.user_template
    assert "Impression" in cfg.user_template


def test_prompt_with_indication() -> None:
    cfg = chest_xray_prompt(indication="Cough and fever")
    assert "Cough and fever" in cfg.user_template
    assert "Findings" in cfg.user_template
    assert "Impression" in cfg.user_template


# ----------------------------------------------------------- extraction tests

def test_extract_sections_standard() -> None:
    text = "Findings: Normal heart size.\nImpression: No acute cardiopulmonary process."
    findings, impression = extract_sections(text)
    assert findings == "Normal heart size."
    assert impression == "No acute cardiopulmonary process."


def test_extract_sections_missing_impression() -> None:
    text = "Findings: Normal heart size."
    findings, impression = extract_sections(text)
    assert findings == "Normal heart size."
    assert impression is None


def test_extract_sections_empty() -> None:
    findings, impression = extract_sections("")
    assert findings is None
    assert impression is None


def test_extract_sections_variations() -> None:
    # All-caps
    f, i = extract_sections("FINDINGS: Normal.\nIMPRESSION: Normal.")
    assert f == "Normal."
    assert i == "Normal."

    # Lowercase, extra space before colon
    f2, i2 = extract_sections("findings: Normal.\nimpression : Normal.")
    assert f2 is not None and "Normal" in f2
    assert i2 is not None and "Normal" in i2


# -------------------------------------------------------------- output tests

def test_format_result(mock_generation_result: GenerationResult, mock_model_info: ModelInfo) -> None:
    rec = format_result(mock_generation_result, "study_001", "/img/a.png", "deadbeef", mock_model_info)
    required_keys = {
        "study_id", "image_path", "image_sha256", "raw_report",
        "findings", "impression", "status", "warnings",
        "model_id", "model_revision", "precision",
        "prompt_version", "generation_config", "elapsed_seconds",
    }
    assert required_keys <= set(rec.keys())
    assert rec["study_id"] == "study_001"


def test_append_jsonl_creates_file(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    append_jsonl({"id": 1}, out)
    assert out.exists()
    assert json.loads(out.read_text(encoding="utf-8").strip()) == {"id": 1}


def test_append_jsonl_appends(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    append_jsonl({"id": 1}, out)
    append_jsonl({"id": 2}, out)
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[1]) == {"id": 2}


# ----------------------------------------------------------- manifest tests

def test_load_manifest_valid(tmp_path: Path) -> None:
    m = tmp_path / "manifest.jsonl"
    m.write_text(
        '{"study_id": "1", "image_path": "a.png"}\n'
        '{"study_id": "2", "image_path": "b.png"}\n',
        encoding="utf-8",
    )
    entries = load_manifest(m)
    assert len(entries) == 2


def test_load_manifest_duplicate_ids(tmp_path: Path) -> None:
    m = tmp_path / "manifest.jsonl"
    m.write_text(
        '{"study_id": "1", "image_path": "a.png"}\n'
        '{"study_id": "1", "image_path": "b.png"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        load_manifest(m)


def test_load_manifest_missing_fields(tmp_path: Path) -> None:
    m = tmp_path / "manifest.jsonl"
    m.write_text('{"study_id": "1"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="image_path"):
        load_manifest(m)


def test_load_completed_empty(tmp_path: Path) -> None:
    assert load_completed(tmp_path / "nonexistent.jsonl") == set()


# -------------------------------------------------------------- batch tests

def test_batch_resume_skips_completed(tmp_path: Path, tiny_image_path: Path) -> None:
    """Verify that resumed batches skip already-completed studies."""
    out = tmp_path / "out.jsonl"
    # Pre-seed one completed record.
    append_jsonl({"study_id": "1", "status": "success"}, out)

    manifest = [
        {"study_id": "1", "image_path": str(tiny_image_path)},
        {"study_id": "2", "image_path": str(tiny_image_path)},
    ]

    # Stub model/processor that won't actually run inference.
    mock_model = MagicMock()
    mock_processor = MagicMock()
    mock_info = ModelInfo("test", None, "float16", "auto", False)

    # Patch generate_report so we don't need real weights.
    from unittest.mock import patch
    fake_result = GenerationResult(
        raw_report="Findings: OK.\nImpression: OK.",
        findings="OK.", impression="OK.",
        status="success", elapsed_seconds=0.1,
        prompt_version="chest_xray_v1",
        generation_config={"max_new_tokens": 384, "do_sample": False},
    )
    with patch("cxr_report.batch.generate_report", return_value=fake_result):
        stats = run_batch(mock_model, mock_processor, mock_info, manifest, out, resume=True)

    assert stats["skipped"] == 1
    assert stats["completed"] == 1
    lines = out.read_text(encoding="utf-8").strip().split("\n")
    # Original seeded record + newly generated record
    assert len(lines) == 2
