"""Model loading for MedGemma inference.

Call ``load_medgemma`` once per process; reuse the returned objects
for all subsequent generation calls.
"""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about the loaded model checkpoint."""

    model_id: str
    revision: str | None
    dtype: str
    device_map: str
    quantized: bool


def load_medgemma(
    checkpoint: str = "google/medgemma-1.5-4b-it",
    device_map: str = "auto",
    quantize_4bit: bool = False,
) -> tuple[Any, Any, ModelInfo]:
    """Load the MedGemma model and processor exactly once.

    Args:
        checkpoint: Hugging Face model ID or local path.
        device_map: Device mapping strategy (``'auto'`` recommended).
        quantize_4bit: Load in 4-bit precision via bitsandbytes.

    Returns:
        ``(model, processor, ModelInfo)``

    Raises:
        ImportError: When ``torch`` or ``transformers`` is missing.
    """
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        raise ImportError(
            "Install the generation extras: pip install -e '.[generation]'"
        ) from exc

    processor = AutoProcessor.from_pretrained(checkpoint)

    load_kwargs: dict[str, Any] = {"device_map": device_map}
    dtype_label = "float32"

    if quantize_4bit:
        from transformers import BitsAndBytesConfig

        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        dtype_label = "4bit"
    elif torch.cuda.is_available():
        load_kwargs["torch_dtype"] = torch.float16
        dtype_label = "float16"

    model = AutoModelForImageTextToText.from_pretrained(checkpoint, **load_kwargs)
    model.eval()

    info = ModelInfo(
        model_id=checkpoint,
        revision=getattr(model.config, "_commit_hash", None),
        dtype=dtype_label,
        device_map=device_map,
        quantized=quantize_4bit,
    )

    return model, processor, info
