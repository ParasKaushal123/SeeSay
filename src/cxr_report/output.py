"""Output parsing and serialization."""

import json
import re
from pathlib import Path
from typing import Any


def extract_sections(text: str) -> tuple[str | None, str | None]:
    """Parse Findings and Impression sections from generated text.

    Handles common heading variations: Findings/FINDINGS/findings,
    with or without colon, with optional whitespace.  Never invents
    a normal report when headings are absent.

    Returns:
        (findings_text, impression_text) — either may be None.
    """
    if not text or not text.strip():
        return None, None

    # Pattern matches "Findings" (case-insensitive) followed by optional
    # whitespace and an optional colon, then captures everything until the
    # next known heading or end-of-string.
    findings_pattern = re.compile(
        r"(?i)\bfindings?\s*:\s*(.*?)(?=\bimpressions?\s*:|$)",
        re.DOTALL,
    )
    impression_pattern = re.compile(
        r"(?i)\bimpressions?\s*:\s*(.*)",
        re.DOTALL,
    )

    findings_match = findings_pattern.search(text)
    impression_match = impression_pattern.search(text)

    findings = findings_match.group(1).strip() if findings_match else None
    impression = impression_match.group(1).strip() if impression_match else None

    # Treat empty captures as absent.
    if findings == "":
        findings = None
    if impression == "":
        impression = None

    return findings, impression


def format_result(
    result: Any,
    study_id: str,
    image_path: str | Path,
    image_hash: str,
    model_info: Any,
) -> dict[str, Any]:
    """Format generation output into a JSON-serializable record.

    Args:
        result: A GenerationResult instance.
        study_id: Unique study identifier.
        image_path: Path to the processed image.
        image_hash: SHA-256 hex digest of the image file.
        model_info: A ModelInfo instance.
    """
    return {
        "study_id": study_id,
        "image_path": str(image_path),
        "image_sha256": image_hash,
        "raw_report": result.raw_report,
        "findings": result.findings,
        "impression": result.impression,
        "status": result.status,
        "warnings": list(result.warnings),
        "model_id": model_info.model_id,
        "model_revision": model_info.revision,
        "precision": model_info.dtype,
        "prompt_version": result.prompt_version,
        "generation_config": result.generation_config,
        "elapsed_seconds": result.elapsed_seconds,
    }


def save_result(result: dict[str, Any], path: str | Path) -> None:
    """Write a single result to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def append_jsonl(result: dict[str, Any], path: str | Path) -> None:
    """Append one JSON-lines record and flush immediately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(result, ensure_ascii=False) + "\n")
        fh.flush()
