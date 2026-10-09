"""Batch processing module for CXR report generation."""

import json
import time
from pathlib import Path
from typing import Any

from .images import image_sha256
from .generator import generate_report
from .output import format_result, append_jsonl
from .prompts import chest_xray_prompt


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """
    Load a JSONL manifest file containing study_id and image_path.
    
    Args:
        path: Path to the JSONL manifest file.
        
    Returns:
        List of parsed dictionaries.
        
    Raises:
        ValueError: If study_id or image_path is missing, or if study_ids are not unique.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    manifest = []
    seen_study_ids = set()
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_num}: {e}")
                
            if "study_id" not in entry:
                raise ValueError(f"Missing 'study_id' at line {line_num}")
            if "image_path" not in entry:
                raise ValueError(f"Missing 'image_path' at line {line_num}")
                
            study_id = entry["study_id"]
            if study_id in seen_study_ids:
                raise ValueError(f"Duplicate 'study_id' found: {study_id}")
                
            seen_study_ids.add(study_id)
            manifest.append(entry)
            
    return manifest


def load_completed(output_path: str | Path) -> set[str]:
    """
    Read an existing output JSONL file to find completed study_ids.
    
    Args:
        output_path: Path to the output JSONL file.
        
    Returns:
        Set of study_ids with status='success'.
    """
    output_path = Path(output_path)
    completed = set()
    
    if not output_path.exists():
        return completed
        
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("status") == "success" and "study_id" in data:
                    completed.add(data["study_id"])
            except json.JSONDecodeError:
                pass
                
    return completed


def run_batch(
    model: Any,
    processor: Any,
    model_info: Any,
    manifest: list[dict[str, Any]],
    output_path: str | Path,
    *,
    resume: bool = True,
    max_new_tokens: int = 384
) -> dict[str, Any]:
    """
    Process studies sequentially and generate reports.
    
    Args:
        model: Loaded MedGemma model.
        processor: Loaded processor.
        model_info: ModelInfo dataclass instance.
        manifest: List of study dictionaries.
        output_path: Path to append output JSONL results.
        resume: Whether to skip already completed studies.
        max_new_tokens: Max new tokens for generation.
        
    Returns:
        Dictionary with run summary statistics.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    completed = set()
    if resume:
        completed = load_completed(output_path)
        
    stats = {
        "total": len(manifest),
        "completed": 0,
        "failed": 0,
        "skipped": 0,
        "elapsed_seconds": 0.0,
    }
    
    start_time = time.time()
    
    for idx, entry in enumerate(manifest, 1):
        study_id = entry["study_id"]
        image_path = entry["image_path"]
        indication = entry.get("indication")
        
        if resume and study_id in completed:
            stats["skipped"] += 1
            print(f"[{idx}/{stats['total']}] study_id {study_id} skipped (already completed)")
            continue
            
        study_start = time.time()
        
        try:
            # Generate prompt
            prompt_config = chest_xray_prompt(indication)
            
            # Generate report
            result = generate_report(
                model=model,
                processor=processor,
                image_path=image_path,
                indication=indication,
                max_new_tokens=max_new_tokens,
                prompt_config=prompt_config,
            )
            
            # Hash image
            img_hash = image_sha256(image_path)
            
            # Format and save result
            formatted = format_result(
                result=result,
                study_id=study_id,
                image_path=image_path,
                image_hash=img_hash,
                model_info=model_info,
            )
            append_jsonl(formatted, output_path)
            stats["completed"] += 1
            
        except Exception as e:
            # Record failure
            error_result = {
                "study_id": study_id,
                "image_path": image_path,
                "status": "error",
                "error": str(e),
                "model_id": model_info.model_id,
                "timestamp": time.time(),
            }
            append_jsonl(error_result, output_path)
            stats["failed"] += 1
            
        study_elapsed = time.time() - study_start
        print(f"[{idx}/{stats['total']}] study_id {study_id} {study_elapsed:.1f}s")
        
    stats["elapsed_seconds"] = time.time() - start_time
    return stats
