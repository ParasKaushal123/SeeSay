# %% [markdown]
# # MedGemma CXR Report Generation — Kaggle Notebook
#
# This notebook generates chest X-ray reports using `google/medgemma-1.5-4b-it`.
# It is designed for Kaggle GPU environments.  Upload the repository to
# `/kaggle/working/radiology` before running.
#
# **Two entry paths:**
# 1. If `model` and `processor` are already loaded in the session, reuse them.
# 2. Otherwise, authenticate and load once.

# %% Cell 1 — Runtime check
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except ImportError:
    print("PyTorch not installed")

try:
    import transformers
    print(f"Transformers: {transformers.__version__}")
except ImportError:
    print("Transformers not installed")

# %% Cell 2 — Install the pipeline package
# Assumes the repository is uploaded to /kaggle/working/radiology
# !pip install -e "/kaggle/working/radiology[generation]"

# %% Cell 3 — Load the model once
import gc
from cxr_report.model import load_medgemma

if "model" not in dir() or "processor" not in dir():
    print("Loading MedGemma (this takes ~60 s on a T4) …")
    model, processor, model_info = load_medgemma(
        checkpoint="google/medgemma-1.5-4b-it",
        device_map="auto",
        quantize_4bit=False,  # set True if running on ≤8 GB VRAM
    )
    print(f"Model loaded — revision {model_info.revision}, dtype {model_info.dtype}")
else:
    print("Reusing existing model/processor objects.")
    from cxr_report.model import ModelInfo
    model_info = ModelInfo(
        model_id="google/medgemma-1.5-4b-it",
        revision=getattr(model.config, "_commit_hash", None),
        dtype=str(next(model.parameters()).dtype),
        device_map="auto",
        quantized=False,
    )

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% Cell 4 — Single-image smoke test
from cxr_report.generator import generate_report
from cxr_report.images import image_sha256
from cxr_report.output import format_result
import json

# Replace with a real IU X-ray path.
sample_path = "/kaggle/input/iu-xray/images/CXR1_1_IM-0001-3001.png"

result = generate_report(
    model, processor, sample_path,
    indication="Routine chest X-ray",
    max_new_tokens=384,
)

print(f"Status: {result.status}")
print(f"Elapsed: {result.elapsed_seconds:.1f}s")
print(f"Warnings: {result.warnings}")
print()
print("--- RAW REPORT ---")
print(result.raw_report)
print()
print(f"Findings:   {result.findings}")
print(f"Impression: {result.impression}")

# Save single result
single = format_result(result, "smoke_test", sample_path, image_sha256(sample_path), model_info)
print(json.dumps(single, indent=2))

# %% Cell 5 — Batch inference over IU X-ray
from pathlib import Path
from cxr_report.batch import load_manifest, run_batch

manifest_path = Path("/kaggle/working/radiology/examples/manifest.jsonl")
output_path = Path("/kaggle/working/reports/iu_reports.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

if manifest_path.exists():
    manifest = load_manifest(manifest_path)
    print(f"Manifest loaded: {len(manifest)} studies")

    stats = run_batch(
        model, processor, model_info,
        manifest, output_path,
        resume=True,
        max_new_tokens=384,
    )

    print(f"\nBatch complete: {json.dumps(stats, indent=2)}")
else:
    print(f"Manifest not found at {manifest_path}")
    print("Create one with:  {\"study_id\": \"...\", \"image_path\": \"...\", \"indication\": \"...\"}")

# %% Cell 6 — Download results
print(f"\nOutput file: {output_path}")
if output_path.exists():
    with open(output_path) as fh:
        lines = fh.readlines()
    print(f"Total records: {len(lines)}")
    # Show first record
    if lines:
        print("\nFirst record:")
        print(json.dumps(json.loads(lines[0]), indent=2))
    # In Kaggle, download from the Output tab or use:
    # from IPython.display import FileLink
    # display(FileLink(str(output_path)))
else:
    print("No output file generated yet.")
