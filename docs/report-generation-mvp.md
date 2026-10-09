# Implementation plan: MedGemma X-ray report generation

Updated: 9 October 2026. This is a build plan; the report-generation modules described here are not implemented yet.

## Starting point and deliverable

The user successfully ran `google/medgemma-1.5-4b-it` in Kaggle with `AutoProcessor`, `AutoModelForMultimodalLM`, `device_map="auto"`, and `model.generate`. The checkpoint is downloaded and the `model` and `processor` objects are loaded in that running session. The candy example confirms that inference works; chest-report behavior still needs a smoke test.

Reuse these existing objects. Keep the working model-loading configuration initially and record its actual precision and device placement. Four-bit loading is an optional memory fallback, not a requirement. No training is needed.

The deliverable is a GitHub-ready pipeline that accepts a chest X-ray, generates Findings and Impression, exports text/JSON, and processes IU images in batches. The existing LightGBM functionality is a later component. The pipeline generates reports from X-rays, rather than generating new X-ray images.

## Architecture

```text
PNG/JPEG X-ray + optional indication
                  |
        validate and open as RGB
                  |
     chest-report prompt + chat template
                  |
       existing loaded MedGemma model
                  |
      decode generated tokens only
                  |
     extract Findings and Impression
                  |
        save text + JSON metadata
```

One image per study is the first supported configuration. Frontal/lateral input follows after testing the single-image path and is recorded as a separate configuration.

## 1. Capture the working runtime

Record Python, PyTorch, Transformers, Accelerate, Pillow, GPU name, model checkpoint revision, dtype, and `hf_device_map`. Confirm computation uses the expected GPU. Export tested dependency versions rather than changing a working environment before the first chest-X-ray test.

The notebook will reuse `model` and `processor` while the session is alive. A fresh session needs to load the checkpoint again. Authentication stays outside code and saved notebook outputs.

**Deliverable:** a runtime-check notebook cell and reproducibility metadata.

## 2. Image input and reporting prompt

Create `src/cxr_report/images.py` to accept a local PNG/JPEG, validate it with Pillow, and convert it to RGB. Preserve orientation. Let the official processor handle expected resizing; do not add arbitrary flips or contrast changes. Missing, corrupt, and unsupported inputs produce explicit errors.

Create `src/cxr_report/prompts.py` with a versioned prompt requesting:

```text
Findings: ...
Impression: ...
```

Ask for visible observations and appropriate uncertainty. Include an indication only when supplied. State that no prior study is provided in single-image mode, so the prompt does not request interval comparisons. Reference reports and diagnosis labels never enter the generation prompt.

**Deliverable:** the candy image and question are replaced by a real chest X-ray and a fixed report prompt.

## 3. Generate with the existing model

Create `src/cxr_report/generator.py` with this planned interface:

```python
generate_report(
    model,
    processor,
    image_path,
    indication=None,
    max_new_tokens=384,
)
```

The function will:

1. Build a message containing the PIL image and chest-report prompt.
2. Use the already-working `processor.apply_chat_template` path with generation prompt, tokenization, return dictionary, and PyTorch tensors enabled.
3. Move inputs to the correct input device for the working configuration.
4. Run `model.eval()` and `torch.inference_mode()`.
5. Generate with `do_sample=False`, batch size one, and a configurable token limit. Start at 384 new tokens and measure truncation.
6. Slice off input tokens and decode with `skip_special_tokens=True`.
7. Record elapsed time, empty-output failures, and possible truncation.

This function receives a loaded model and must not reload weights per image. Add an optional loader in `src/cxr_report/model.py` for standalone CLI usage, using the tested API/configuration and loading once per process.

**Deliverable:** one real chest X-ray produces a nonempty generated report. Record actual inference latency and peak memory.

## 4. Parse and save reports

Create `src/cxr_report/output.py`. Preserve the complete raw response and extract Findings/Impression when headings exist. Handle simple heading variations. Missing headings produce a parsing warning and nullable sections, never an invented normal report.

Each JSON result will include:

| Field | Content |
|---|---|
| `study_id`, `image_path`, `image_sha256` | Input identity. |
| `raw_report`, `findings`, `impression` | Generated text and parsed sections. |
| `status`, `warnings` | Failures, parsing issues, or possible truncation. |
| `model_id`, `model_revision`, `precision` | Actual model configuration. |
| `prompt_version`, `generation_config` | Reproducibility metadata. |
| `elapsed_seconds` | Measured inference time. |

Save text and JSON under `/kaggle/working/reports/`. Download/preserve successful outputs before ending the session.

**Deliverable:** downloadable reports traceable to the input image and generation configuration.

## 5. IU batch inference

Create `src/cxr_report/batch.py`, accepting JSONL records such as:

```json
{"study_id":"example_001","image_path":"/kaggle/input/iu-images/example.png","indication":"Chest pain"}
```

Validate IDs and paths. Load the model once, process studies sequentially, and append/flush one result per study. Track completed and failed counts. Keep explicit error records rather than silently omitting studies.

Resume skips only successful records matching the same image hash and generation configuration. Changed models/prompts trigger regeneration; failed records can be retried. Do not accept reference-report fields as inference input.

Start with five development images, then 20–50 IU images. Estimate full-batch runtime from measured seconds per image. Reserve the official test split for later frozen evaluation; use training/development images for prompt and runtime changes. CheXpert Plus comes later.

**Deliverable:** resumable IU smoke batch and a run summary.

## 6. Notebook, CLI, and repository package

Planned files:

```text
src/cxr_report/
  __init__.py
  model.py
  images.py
  prompts.py
  generator.py
  output.py
  batch.py
  cli.py
  __main__.py
notebooks/medgemma_report_generation.ipynb
examples/manifest.jsonl
tests/test_report_pipeline.py
```

The Kaggle notebook will support two entry paths: reuse existing `model`/`processor`, or authenticate and load once in a fresh session. Both use the same generation functions.

Planned commands, available after implementation:

```text
python -m cxr_report generate --image IMAGE.png --output report.json
python -m cxr_report batch --manifest input.jsonl --output reports.jsonl --resume
```

Add a generation dependency extra in `pyproject.toml`. Write a concise README covering the tested Kaggle setup, model access, image command, batch manifest, output example, model attribution, and measured resource use.

Exclude credentials, weights, model caches, and image datasets from Git. Keep notebook outputs free of secrets. Create a report-generation commit after checks pass. Configure the destination repository before a remote push. Published model benchmark scores remain separate from this project's measurements.

## 7. Verification and completion criteria

- Unit tests cover bad inputs, section extraction, output serialization, batch errors, and resume/configuration matching.
- Mocked model tests exercise orchestration in CI without downloading weights.
- One real Kaggle X-ray inference produces a report using the existing model.
- A small IU batch succeeds without repeated loading and records latency, memory, failures, and truncation.
- Text/JSON outputs can be downloaded from Kaggle.
- The README and notebook commands match implemented behavior.

The initial milestone requires no training or evidence-fusion integration. Following milestones measure the unedited IU baseline, add evidence-based correction, and evaluate on CheXpert Plus.

## Estimated work

| Work | Planning estimate |
|---|---:|
| Core generation, image handling, parsing, export | 1–2 hours |
| Batch/resume and CLI | 1–2 hours |
| Notebook, tests, README, packaging | 1–2 hours |
| First runnable packaged pipeline | Approximately 3–6 hours, depending on compatibility fixes |

The successful Kaggle load removes the initial download step for the current session. Per-image time must be measured on the allocated GPU; earlier latency estimates are provisional.

Reference: [official MedGemma model card](https://huggingface.co/google/medgemma-1.5-4b-it).
