# CXR evidence pipeline — initial implementation

This first version trains **small logistic finding probes, a shallow LightGBM evidence model, and held-out sigmoid calibration** from cached study features. It runs on CPU. It does not download models, read X-ray pixels, generate reports, or claim benchmark results yet.

The project is IU X-ray first, for a laptop with 16 GB RAM and an RTX 5050 with approximately 8 GB VRAM. CheXpert Plus and ReXGradient are later evaluation stages. MIMIC-CXR is not a required project dataset.

## Run the initial version

PowerShell, from this directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m cxr_evidence demo --output artifacts/demo
.\.venv\Scripts\python.exe -m pytest
```

The demo creates a **synthetic** fixture with 240 studies, grouped patients, three findings, and simulated evidence. Its metrics validate plumbing, not medical accuracy. Reusing a nonempty output directory fails rather than overwriting a previous run.

Once real cached features exist:

```powershell
.\.venv\Scripts\python.exe -m cxr_evidence train --data datasets/iu_features.npz --output artifacts/iu_v1
.\.venv\Scripts\python.exe -m cxr_evidence predict --data datasets/iu_inference.npz --model artifacts/iu_v1/model.joblib --output artifacts/iu_predictions.npz
```

Outputs are `model.joblib`, `metrics.json`, and a prediction archive containing per-finding probabilities. The metrics include Brier score, log loss, and AUROC/average precision where both classes exist. No GREEN score is computed in this commit. Only load trusted local joblib artifacts; they use pickle serialization.

## Cached feature contract

Store the following arrays in one `.npz` using `numpy.savez_compressed`. Use Unicode string arrays, never object arrays. N is studies, C is findings, D is embedding width, E is supplied evidence features.

| Array | Shape / values | Meaning |
|---|---|---|
| `study_ids` | `[N]` strings | Unique studies; aggregate frontal/lateral views before exporting. |
| `patient_ids` | `[N]` strings | Grouping identifier; a patient cannot span splits. If only study IDs are available, document that limitation rather than asserting verified patient separation. |
| `splits` | `[N]` strings | `train`, `calibration`, optional `validation` and `test`; `inference` only for prediction. |
| `image_features` | `[N,D]` finite float32 | Frozen encoder features available at inference. |
| `evidence` | `[N,C,E]` float32 | Additional finding-specific features; NaN means unavailable, not negative. |
| `finding_names` | `[C]` unique strings | Identical target order across runs. |
| `evidence_names` | `[E]` unique strings | Names start with `healthy__`, `regional__`, or `reader__`. E may be zero. |
| `labels` | `[N,C]` int8 | 1 positive, 0 explicitly negative, -1 unknown/uncertain/unmentioned. Optional for prediction. |
| `label_source` | scalar string | `report_derived`, `image_adjudicated`, or `synthetic`. |
| `encoder_id` | scalar string | Encoder checkpoint revision plus preprocessing/pooling identity. |

Do not label unmentioned findings as absent. “No acute disease” does not establish absence of every chronic finding. Labels, original reference text, and metric scores must never appear among inference features.

Cached external evidence must come from frozen, untuned sources, or be generated out of fold. This initial implementation cross-fits its own image probes; it cannot detect leakage inside externally supplied evidence. Build normal-reference banks using training patients only and exclude the query patient. Keep separate archives if label provenance differs; the initial format has one label source per archive.

## Training design

1. Validate shapes, identifiers, split separation, label states, and feature identity.
2. On the training split, use three patient-grouped folds. Fit standard scaling and one regularized logistic probe per finding on each fold's known labels; predict the excluded patients. A fold with one class uses a smoothed training prior.
3. Train one shared shallow LightGBM model on the resulting out-of-fold probe scores, supplied evidence, missingness indicators, and categorical finding identity. Unknown targets are excluded. Default: 80 trees, depth 3, seven leaves, two CPU threads.
4. Refit probes on all training patients. Freeze probes and LightGBM.
5. Fit one pooled sigmoid calibrator on the separate calibration split. It requires positive and negative labels. Pooling supports IU's limited sample size; it does not establish calibration for each finding or new hospital.
6. Evaluate untouched validation/test splits and serialize the model, configuration, dataset SHA-256, package versions, class counts, and timing.

Do not tune repeatedly on the test scores written by the tool. This commit does not perform hyperparameter search or set automatic report-edit thresholds. Calibration against report-derived labels measures agreement with those labels; image-based correctness needs independently reviewed images.

## Time and storage

Planning estimates for the complete local project, not measured throughput:

| Work | Expected scale |
|---|---|
| Train probes + LightGBM + calibration on IU-sized cached features | Minutes to a few hours; CPU, no GPU training required. |
| First 50-study model timing run | Measure before scheduling the dataset. |
| Generate all IU reader outputs and regional features | Roughly 2–7 days of GPU time; highly sensitive to reader count, tokens, crop count, and quantization. |
| Lightweight code, environment, small caches | Reserve 5–10 GB. |
| First image/model pipeline | Reserve 60 GB total free disk. |
| Keep UniRG, a second reader, and GREEN locally | Reserve 100–120 GB total free disk. |

Model repository storage reported by Hugging Face on 9 October 2026: MedGemma 1.5 4B approximately 8.64 GB; UniRG-CXR 17.55 GB; GREEN-radllama2-7b 13.48 GB. These are repository totals, not runtime VRAM or a promise of exact downloaded bytes. Selecting files can reduce downloads. Loading in four-bit reduces runtime weight memory but does not necessarily reduce the original checkpoint download. Avoid keeping duplicate caches.

The synthetic demo prints actual `training_seconds` for this lightweight implementation. This is not an estimate of X-ray feature extraction speed.

## Next implementation stages

The full architecture, staged tasks, benchmark protocol, and acceptance criteria are in [the implementation plan](docs/implementation-plan.md). The existing root README and `research/` documents remain literature material; older usage examples there describe proposals, not the current runnable package.
