# IU-first implementation plan

Updated 9 October 2026. Target machine: RTX 5050 Laptop GPU, 8,151 MiB reported VRAM, 16 GB system RAM. Scope: an evidence-checked chest X-ray reporting pipeline, developed on IU X-ray; CheXpert Plus and ReXGradient follow later. No MIMIC download or training is required.

## Architecture and current boundary

```text
Study images + permitted clinical context
    |
    +-- Primary reader ------------------------------ draft report
    +-- Frozen image encoder --> finding probes ----- model features
    +-- Training-only normal reference bank ---------- reference distances
    +-- Fixed regional scans + selective crop reader - regional evidence
    +-- Separate reader ----------------------------- finding assertions
                          |
                  shallow LightGBM
                          |
                  held-out calibration
                          |
                  conservative edit policy
                          |
              span edits + consistency validation
                          |
                 corrected report + audit log
```

The initial commit implements training and prediction from cached features, including the finding probes, LightGBM, calibration, and metrics. Image preprocessing, model inference, retrieval, regional scanning, report parsing, and report editing are subsequent stages. The synthetic fixture exercises interfaces without supplying medical evidence.

Use real reference images. Their similarity is one imperfect feature, not proof of health. Separate readers may have correlated errors; the system does not assume statistical independence. Keep the unedited report as a candidate and fallback. Full four-channel and best-performing ablated versions must be reported separately if a channel hurts performance.

## 1. Hardware and model smoke tests

Implement `readers/base.py`, `readers/medgemma.py`, and `readers/unirg.py` with one interface: study in, findings/impression plus model revision and generation metadata out.

Start with [MedGemma 1.5 4B](https://huggingface.co/google/medgemma-1.5-4b-it) in four-bit mode. Test [UniRG-CXR](https://huggingface.co/microsoft/UniRG-CXR) separately in four-bit mode with batch size one. UniRG's published example uses a fixed 262,144-pixel budget and single image; reproduce that path before experimenting with extra views. Its published configuration is not a guarantee that quantized inference fits 8 GB. If it fits reliably, retain UniRG as the primary draft candidate; otherwise the 4B implementation remains the working local path.

Use Transformers first, SDPA where supported, one resident model, and short bounded outputs initially. Do not preload several readers or the evaluator. Freeze model revision, prompt, processor settings, quantization, context length, and output limit. Test checkpoint loading as well as inference: CPU checkpoint conversion can exceed 16 GB even when quantized weights fit VRAM. Use sharded loading without retaining a second model copy. Confirm the installed PyTorch/CUDA build supports the GPU before a long run. [Quantization documentation](https://huggingface.co/docs/transformers/quantization/bitsandbytes)

**Deliverable:** a 50-study development-only timing report with load time, peak CPU/GPU memory, median/p95 seconds per study, failures, and truncations. The complete project runtime is computed from those measurements:

`hours = (N × reader_seconds + crop_calls × crop_seconds + evaluation_calls × judge_seconds) / 3600`

Include retry overhead. The 2–7 GPU-day estimate in IMPLEMENTATION.md is provisional. Do not schedule full inference until the smoke test passes.

## 2. IU data adapter and immutable splits

Implement `datasets/iu.py`, `datasets/splits.py`, and a manifest builder. Obtain IU images/reports from [Open-i](https://openi.nlm.nih.gov/) or the source linked by the [R2Gen authors](https://github.com/zhjohnchan/R2Gen), then reconcile original IDs and image paths with the benchmark manifests. Inspect actual resolution: resized downloads cannot provide native-resolution evidence.

The current ReXrank IU manifests contain **296 validation studies and 590 test studies**, verified directly during planning. The JSON is keyed by study ID, and `context` is a string in the inspected manifest, despite the general README describing structured context. Parse the actual files, including the singular `report` key. Support optional metadata rather than assuming age, sex, or AP/PA labels exist. [Official manifest directory](https://github.com/rajpurkarlab/ReXrank/tree/main/data/iu_xray)

Reserve the 590 test IDs before constructing training data, normal banks, or labels. Reconcile the training collection against those IDs and the 296 validation IDs; do not assume a mirror's split matches. Keep study views together and enforce patient groups wherever genuine patient identifiers exist. Record the limitation if identifiers only distinguish studies. Detect exact duplicate images across splits; investigate near duplicates without silently altering the official test set.

Divide the official validation patients into calibration and policy-selection portions using a fixed seed and grouped splitting, approximately half each. The initial CLI calls them `calibration` and `validation`. Keep target ontology/hyperparameter development inside training data, with a small prespecified set of policy variants. IU's small validation set does not justify reliable per-disease calibration of rare findings.

Use separate inference manifests containing only images, IDs, views, and permitted context. Store reference reports separately for labels/evaluation. Do not expose reference findings or impressions to readers, retrieval, or the editor.

**Acceptance:** reproducible manifest hashes, zero known group overlap, ID/image integrity checks, and all official test studies accounted for. No silent removal of failed cases.

## 3. Baseline reports, target vocabulary, and parsing

Cache one deterministic report per study/model before training the correction layer. Run each model over its entire batch before unloading it. Write resumable JSONL records with the image hash, prompt hash, model revision, status, and output sections. A failed generation is a recorded failure, not an empty normal report.

Implement `reports/schema.py` and `reports/parser.py`. Store each claim's finding, polarity, uncertainty, laterality, location, source sentence/span, and source model. Begin with standard CXR findings; use training-only counts to decide which can support automatic edits. Add IU-relevant observations such as calcified granulomas or scarring only when parsing and labels are reliable. Preserve unsupported target classes as unedited text. Avoid pretending that a fixed vocabulary covers every abnormality.

Use a negation-aware text extractor or an established clinical parser, with a small manually checked report-only fixture. Text annotation can verify extraction fidelity but cannot establish image truth. Explicit absence, uncertainty, and unmentioned must remain distinct. Rare classes without both training classes are rejected by the initial trainer.

**Acceptance:** parser cases cover coordinated negation, uncertainty, bilateral findings, temporal claims, and multiple findings in a sentence. Establish the uncorrected local baseline before editing. Do not infer clinical accuracy from report-label agreement alone.

## 4. The four evidence channels

| Channel | Initial implementation | Safeguards and cost |
|---|---|---|
| Model features | Cache a frozen [RAD-DINO](https://huggingface.co/microsoft/rad-dino) study representation, then train logistic finding probes. Optional UniRG vision-feature probes come later. | Follow the encoder processor. Pool views consistently and record the encoder/preprocessing identity. Do not probe reference-conditioned decoder states. Small CPU training after extraction. |
| Healthy references | Retrieve a small bank of training studies with explicit normal evidence; compute embedding-distance summaries and match-quality features. | “No acute disease” alone is not a healthy-bank label. Exclude query patients and all held-out patients. Match view first; use demographics only when available. No pixel subtraction between unrelated patients. An empty/poor bank produces missing features. |
| Regional scans | Begin with a fixed, reproducible anatomical crop grid and cheap regional embeddings; add a crop reader for uncertain or discordant findings after measuring cost. | Preserve crop coordinates and orientation. Keep global context available. Do not assume crop-trained performance from a full-image model. Compare with full-image-only evidence. No unvalidated quantitative measurements. |
| Separate readings | Cache a second model's report and parse its positive, negative, uncertain, and unmentioned assertions. | Do not show the primary draft to the second reader. Never map unmentioned to negative. Report disagreement, missingness, and model identity. Start with two total readers. |

The lightweight trainer consumes the feature contract in IMPLEMENTATION.md. For bank-based training features, use reference pools that exclude the query and its patient. If learned retrieval, crop heads, or reader adaptation is added, generate those features out of fold as well. Frozen third-party pretraining exposure remains a separate disclosure; these local splits cannot prove the backbone never saw IU.

Do a cheap systematic scan over the supported finding vocabulary, including findings absent from every report. Otherwise the system can delete errors but cannot discover omissions. Invoke expensive crop questions only with a frozen routing policy, and generate training/calibration features using the same routing policy used at inference. Record missing scans explicitly.

**Acceptance:** reproducible feature caches, no reference text among input features, valid missingness flags, and an ablation showing the incremental value of each channel. Keep healthy references as an implemented optional channel if it does not improve held-out results.

## 5. Train evidence fusion and define calibrated edits

The initial implementation supplies patient-grouped out-of-fold probes, shared shallow LightGBM, and pooled sigmoid calibration. Add a regularized logistic stacker as the mandatory low-capacity comparison. Keep the hyperparameter search small; optimize on development data only. Calibration against weak report labels must be labeled accordingly. Image-adjudicated labels would improve confidence estimates but are not assumed available for the first runnable build.

Next implement `policy.py` with separate actions: keep, add, remove, or abstain from editing. Tune a small number of global thresholds on the policy split before considering per-finding thresholds. An unknown score or poor-quality crop defaults to no automatic edit. Avoid a whole-study normal gate: one missed abnormality would erase useful content.

Add a finding only when it is supported by the fused evidence and its validation precision is adequate. Remove an existing claim only when evidence supports its absence, not because another model failed to mention it. Preserve original uncertain wording unless the calibrated evidence supports changing it. Finding presence alone does not authorize new laterality, severity, measurements, or temporal change.

Rare findings without enough reviewed examples remain no-auto-edit targets. Choose eligibility based on sample counts and uncertainty intervals, not a single F1 maximum. The system can still expose their evidence for inspection.

**Acceptance:** improved development edit precision/recall versus the unedited baseline and simple voting, with explicit counts of false additions, deleted true findings, recovered omissions, and untouched claims. Report uncertainty rather than assuming pooled calibration transfers to every disease.

## 6. Minimal report editing and audit output

Implement `reports/editor.py` using validated span-level edits and controlled short clauses first. Split or rewrite only the affected clause when a sentence contains both correct and incorrect findings. Resolve contradictions with existing negative statements. Preserve unrelated observations.

If a language-model rewrite is later used, constrain it to an approved claim list and reject any added finding/attribute. Reparse the report after editing and compare the result with the approved claims. On validation failure, retain the original report and log the failed edit. Keep Findings and Impression consistent when both are generated. Avoid introducing comparisons without actual permitted prior evidence.

Audit records include the original and final reports, each proposed edit, calibrated score, evidence availability, crop coordinates, reference IDs, and validation status. Probabilities, model names, and internal implementation details belong in the audit record rather than clinical prose.

**Acceptance:** tests for contradictory negatives, mixed sentences, duplicate findings, unsupported modifiers, missing evidence, parser disagreement, and fallback behavior. Preserve a valid report for every study.

## 7. Benchmark protocol and success criteria

Primary first-stage evaluation: **IU Findings**, using the official 590-study test manifest. Freeze the chosen model and thresholds before evaluating. Findings + Impression is a separate track. Compare full-view and frontal-only methods with their actual available information clearly stated.

Evaluate paired unedited and corrected outputs with GREEN, RadGraph, and finding extraction metrics. Add CRIMSON/RaTEScore when their exact supported runtimes are integrated. Retain per-study scores and calculate paired bootstrap intervals, grouping by patient where possible. Repeated optimization after seeing test results would turn that test into development data.

Use the [official GREEN implementation](https://github.com/Stanford-AIMI/GREEN) in a separate evaluator process. Its 7B model can itself strain 8 GB VRAM. If quantized local evaluation is necessary, record it as an approximate evaluator configuration; a leaderboard-equivalent claim requires the official evaluator setup or organizer evaluation. Do not replace GREEN with a similarly named heuristic.

Run these comparisons: primary reader; second reader; simple voting; probe-only correction; logistic fusion; LightGBM fusion; full system; each channel removed. Compare edit frequency and useful finding retention so a score gain from overly terse reports is visible. Initially seek a positive paired GREEN difference without a material loss in finding recall; a +0.01 to +0.02 gain is a planning target, not a promised outcome or an achieved result. If the evidence layer fails to beat the stronger unedited local baseline, retain that baseline as the best configuration.

The leaderboard currently lists **0.469 GREEN for UniRG-CXR on private ReXGradient Findings**, and **0.664 on IU Findings**. These are published baseline scores, not results of this project. Four-bit local inference and different prompts can produce different numbers. [ReXrank](https://rexrank.ai/)

## 8. CheXpert Plus, ReXGradient, and release

After freezing IU development, evaluate the IU-trained correction layer on a prespecified CheXpert Plus test set with no target tuning. If subsequently adapting to CheXpert Plus, create a separate version using only its training/calibration partitions and retain an untouched test set. Record institutional shift and per-finding calibration changes. Obtain data through the [official CheXpert Plus page](https://aimi.stanford.edu/datasets/chexpert-plus).

The public ReXGradient-160K test split and the private ReXrank ReXGradient test set are distinct. Do not report a public-split score as the private leaderboard result. Access and use of the public data follow its [dataset terms](https://huggingface.co/datasets/rajpurkarlab/ReXGradient-160K). The first IU version does not need this download.

Package the frozen model revisions, CPU artifacts, manifests, inference environment, and evaluation configuration. The official submission interface is:

```text
python inference.py --input_json_file INPUT --save_json_file OUTPUT --img_root_dir IMAGES
```

Implement it according to the current [submission guidelines](https://rexrank.ai/explore/submission_guideline.html). Build and locally validate the submission bundle; sending it to organizers is a separate action. Support resume, errors, deterministic seeds, and reference-free input. Publish aggregate results and examples only where data terms permit.

## Delivery order

| Milestone | Estimated engineering time | Exit condition |
|---|---|---|
| Initial cached-feature training commit | Current increment | Tests, synthetic smoke run, reusable CPU artifacts. |
| IU adapter + one reader + evaluator smoke tests | 3–5 working days | Verified splits and 50-study resource measurements. |
| Baseline caching + parser + frozen image features | 3–5 working days plus GPU runs | Reproducible baseline and real feature archive. |
| Reference bank + regional scans + second reader | 4–7 working days plus GPU runs | Four implemented evidence interfaces with ablations. |
| Edit policy + writer + validation | 3–5 working days | Conservative end-to-end corrected reports. |
| Frozen IU evaluation + documentation | 2–4 working days plus evaluator runs | Paired metrics, uncertainty, limitations, reproducible artifacts. |
| CheXpert Plus / private ReXrank | Later phase | Access, independent testing, and organizer evaluation where needed. |

Allow roughly 3–6 weeks for the complete IU project, depending on available working hours, model compatibility, and annotation effort. Local GPU inference time is separate from CPU training time. No large-model fine-tuning is required for the first full pipeline; consider it only after the evidence-correction baseline is measured.
