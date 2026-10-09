"""Core report generation using a loaded MedGemma model."""

import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .images import load_image
from .output import extract_sections
from .prompts import PromptConfig, chest_xray_prompt


@dataclass(frozen=True)
class GenerationResult:
    """Immutable result of a single report generation call."""

    raw_report: str
    findings: str | None
    impression: str | None
    status: str  # success | empty_output | truncated | error
    warnings: tuple[str, ...] = ()
    elapsed_seconds: float = 0.0
    prompt_version: str = ""
    generation_config: dict[str, Any] = field(default_factory=dict)


def generate_report(
    model: Any,
    processor: Any,
    image_path: str | Path,
    indication: str | None = None,
    max_new_tokens: int = 384,
    prompt_config: PromptConfig | None = None,
) -> GenerationResult:
    """Generate a chest X-ray report for a single image.

    The model and processor must already be loaded (see ``model.py``).
    This function never reloads weights.

    On any error during generation the function returns a result with
    ``status='error'`` rather than raising, so batch callers can
    continue processing remaining studies.
    """
    import torch

    start = time.perf_counter()
    warnings: list[str] = []
    status = "success"

    if prompt_config is None:
        prompt_config = chest_xray_prompt(indication=indication)
    # Honour caller's max_new_tokens override.
    effective_max = max_new_tokens if max_new_tokens != 384 else prompt_config.max_new_tokens

    gen_config: dict[str, Any] = {
        "max_new_tokens": effective_max,
        "do_sample": False,
    }

    try:
        image = load_image(image_path)

        # Build the chat-template message list expected by MedGemma.
        messages = [
            {"role": "system", "content": prompt_config.system_message},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_config.user_template},
                ],
            },
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )

        # Move every tensor to the model's device.
        device = next(model.parameters()).device
        inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

        input_len = inputs["input_ids"].shape[1]

        model.eval()
        with torch.inference_mode():
            outputs = model.generate(**inputs, **gen_config)

        generated_tokens = outputs[0][input_len:]
        raw_report = processor.decode(generated_tokens, skip_special_tokens=True)

        # Detect truncation.
        if len(generated_tokens) >= effective_max:
            status = "truncated"
            warnings.append(f"Output truncated at {effective_max} tokens")

        # Detect empty output.
        if not raw_report.strip():
            status = "empty_output"
            warnings.append("Model produced empty output")

        findings, impression = extract_sections(raw_report)

    except Exception as exc:
        status = "error"
        raw_report = ""
        findings = None
        impression = None
        warnings.append(f"Error during generation: {exc}\n{traceback.format_exc()}")

    elapsed = time.perf_counter() - start

    return GenerationResult(
        raw_report=raw_report,
        findings=findings,
        impression=impression,
        status=status,
        warnings=tuple(warnings),
        elapsed_seconds=elapsed,
        prompt_version=prompt_config.version,
        generation_config=gen_config,
    )
