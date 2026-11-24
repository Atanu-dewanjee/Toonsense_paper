# ToonSense

**Robust multimodal understanding of cartoon-driven Bangla–English code-mixed memes.**

Adrita Biswas, Atanu Dewanjee, Hrittika Dutta — School of Science, Engineering and Technology, East Delta
University, Chattogram, Bangladesh. Supervised by Antu Chowdhury.

This repository contains the code, results and paper of ToonSense ([project proposal](docs/ToonSense_Proposal.pdf)).
It benchmarks text-only, image-only and multimodal models on Bangla memes: CMBan (humour, sarcasm, offensiveness,
motivation, overall sentiment), BanglaAbuseMeme (abuse, vulgarity, sarcasm, sentiment) and MemoSen (sentiment). It
then measures how stable those models are under controlled linguistic and visual perturbations, tests a
Romanisation-augmentation mitigation, explains the decisions before and after each perturbation, and categorises the
failures. Each dataset is processed separately with the same protocol.


## Repository layout

```
configs/    experiment configuration (datasets, encoders, heads, perturbations)
src/        toonsense library: data, OCR, encoders, heads, CV engine, perturbations, metrics, explainability
scripts/    pipeline steps 01–16 (see table below)
tests/      unit tests
results/    per-dataset scores, tables and figures, and RESULTS.md
paper/      LaTeX sources, generated tables and figures
docs/       project proposal
```

Datasets, cached embeddings, model weights and per-meme artefacts that contain dataset text or meme images are not
included; they are recreated by running the pipeline (see [Data note](#data-note)).

## Pipeline

For the full design (model architecture, heads, evaluation, perturbations, explainability), see
[ARCHITECTURE.md](ARCHITECTURE.md).

```
CMBan images ─► OCR (EasyOCR bn+en) ─► text ──► frozen text encoder (XLM-R, …) ─┐
            └─────────────────────────────────► frozen image encoder (CLIP, …) ─┴► fusion head ─► 5 tasks
                     │                                    │
          linguistic perturbations              visual perturbations
                     └──────────► Robustness Drop / flip rate ◄──────────┘
                                   │
                    IG (modality + token) · Grad-CAM(++) · error categories
```

| Step | Script | Output |
|---|---|---|
| 1 | `scripts/01_ocr.py` | `data/ocr/<dataset>/json/*.json` (text, boxes, confidences), `data/ocr/<dataset>/texts.csv` |
| 2 | `scripts/02_extract_features.py` | `features/<dataset>/{text,image}/<encoder>/clean.npz` |
| 3 | `scripts/03_benchmark.py` | `results/<dataset>/benchmark/{raw_scores,summary,significance}.csv`, OOF probabilities |
| 4 | `scripts/04_robustness.py` | `results/<dataset>/robustness/{perturbed_texts,raw,summary}.csv`, per-variant predictions, example images |
| 5 | `scripts/05_explain.py` | `results/<dataset>/explain/` modality shares, Grad-CAM statistics and figures, token-IG HTML |
| 6 | `scripts/06_error_analysis.py` | `results/<dataset>/error_analysis/` error and fragility tables, manual annotation sheet |
| 7 | `scripts/07_report.py` | `results/<dataset>/tables/*.tex`, `results/<dataset>/figures/*.{png,pdf}` |
| 8 | `scripts/08_flow_diagram.py` | `paper/figures/methodology_flow.{pdf,png}` |
| 9 | `scripts/09_kaggle_package.py` | `kaggle/<dataset>/` self-contained Kaggle GPU kernel that runs steps 1-7 |
| 10 | `scripts/10_cross_dataset.py` | `paper/tables/cross/*.tex`, `paper/figures/cross_robustness.{pdf,png}` |
| 11 | `scripts/11_augmentation.py` | `results/<dataset>/augmentation/` Romanisation-augmentation mitigation experiment |
| 12 | `scripts/12_flow_vertical.py` | `paper/figures/methodology_flow_vertical.{pdf,png}` (proposal-style flowchart) |
| 13 | `scripts/13_encoder_table.py` | `paper/tables/cross/text_encoders.tex` (accuracy and macro-F1 of every text encoder, all datasets) |
| 14 | `scripts/14_kaggle_paper_build.py` | `kaggle/paper-build/` Kaggle CPU kernel that compiles the three LaTeX documents |
| 15 | `scripts/15_flatten_paper.py` | single-file `paper/ToonSense_*.tex` and an Overleaf zip (not versioned) |
| 16 | `scripts/16_conference_tables.py` | `paper/tables/conf/*.tex` compact tables for the conference version |

Every script takes `--dataset {cmban,banglaabusememe,memosen}` (default: `dataset` in `configs/default.yaml`, or
`$TOONSENSE_DATASET`). Run everything with `bash run_all.sh`. Every step caches its outputs, so a re-run picks up where it stopped. The
whole pipeline runs on a CPU. Models are downloaded to `.cache/hf` (about 9 GB for the full encoder set).

## Experimental protocol

* **Inputs.** CMBan memes carry their text inside the image, so the text modality comes from Bangla+English OCR.
  The `data/transcripts_manual_<dataset>.csv` file (`id,text`), if present, overrides OCR with gold transcripts.
  MemoSen ships human-written captions, which are used instead of OCR.
* **Images.** Images are letterboxed to a square before CLIP/DINOv2 preprocessing. The default centre crop would
  otherwise cut the first and last panels off tall multi-panel comics.
* **Encoders** (frozen, embeddings cached).
  * Text: XLM-R base/large, mBERT, BanglaBERT, LaBSE, multilingual-E5.
  * Image: CLIP ViT-B/32, B/16, L/14, DINOv2.
* **Heads.**
  * Unimodal: logistic regression (C chosen on the inner validation split) and an MLP.
  * Fusion: early concat (LR and MLP), Gated Multimodal Unit, bilinear interaction, late (decision) fusion.
  * Baselines: majority class, and character n-gram TF-IDF with LR.
* **Evaluation.**
  * Stratified 5-fold CV repeated over 3 seeds. Early stopping uses an inner stratified validation split carved
    from each training fold, so test folds are never used for model selection.
  * Metrics are computed on out-of-fold predictions: accuracy and macro precision, recall and F1, reported as
    mean ± std over seeds.
  * 95% bootstrap CIs are computed on the seed ensemble. Fusion is compared with the best unimodal model using a
    paired bootstrap test with Holm correction across tasks.
  * Every task is reported at both the original (fine) label granularity and a collapsed (coarse) one.
* **Robustness.**
  * Each fold model is trained once on clean data and then evaluated on clean and perturbed copies of its own
    test fold.
  * Robustness Drop = F1<sub>clean</sub> − F1<sub>pert</sub>, reported in absolute points and relative %. We also
    report the prediction flip rate, which does not need labels.
  * Results are broken down by language group (Bangla vs code-mixed).
  * Decoupled setting: only one modality is perturbed. The optional coupled setting (`--coupled blur jpeg …`)
    re-OCRs the perturbed image.

| Linguistic (severity 1/2/3) | Visual (severity 1/2/3) |
|---|---|
| `translit`: Bangla→colloquial Banglish for 33/66/100% of words | `blur`: Gaussian σ = 1/2/4 px at 224-px scale |
| `roman_noise`: full romanisation + spelling variants on 15/30/50% of words | `crop`: random crop keeping 85/70/55% of the area |
| `bangla_noise`: Bangla orthographic confusions on 10/20/35% of words | `resize`: down-sample to 160/96/48 px, then up-sample back |
| `codemix`: lexicon Bangla↔English switching, p = .33/.66/1 | `jpeg`: quality 30/15/5 |
| `emoji_add`: insert 1/3/6 emojis | `text_occlusion`: hide 33/66/100% of OCR text boxes |
| `emoji_remove`: remove emojis → emoticons → all symbols | `region_occlusion`: grey out 10/25/40% of the non-text area |

All perturbations are deterministic per (meme, family, severity). `perturbed_texts.csv` records how much of each
text actually changed.

* **Explainability.**
  * Integrated Gradients on the fusion-head inputs gives each modality's share of the attribution for every meme,
    before and after perturbation.
  * Token-level IG through XLM-R is run on memes whose prediction flips.
  * Grad-CAM/Grad-CAM++ on CLIP ViT, hooked at the input of the last block, measures (i) how much visual attention
    falls on rendered meme text compared with its area share, and (ii) how stable the CAM is under each
    perturbation.
* **Error analysis.**
  * Fusion errors are labelled linguistic, visual, cross-modal, OCR-limited, or contextual/cultural *candidate*,
    by contrasting the unimodal models trained on the same folds.
  * Robustness failures are labelled linguistic-, visual- or cross-modal-fragile.
  * `manual_annotation_sheet.csv` lets annotators confirm the cultural cases.

## Data note

| Dataset | Memes | Text source | Tasks | Where |
|---|---|---|---|---|
| CMBan | 133 labelled (of 2,641) | EasyOCR | 5 (coarse + fine labels) | `data/raw/` ([CMBan release](https://github.com/newazbenalam/CMBAN), MIT) |
| BanglaAbuseMeme | 4,043 | EasyOCR | 4 | `data/external/banglaabusememe/` (Zenodo, CC-BY-4.0) |
| MemoSen | 4,368 | human captions | 1 | `data/external/memosen/` |

The supplied CMBan archive contains 1,347 images (1,278 unique), but `sample.xlsx` labels only **133** memes (IDs
6578–6714); the full label set is not public. Every script runs unchanged on the full label file: set the `labels`
path of the `cmban` entry in `configs/default.yaml`. With N = 133 the confidence intervals are wide, which is why
the protocol relies on repeated CV, bootstrap CIs and label-free flip rates, and why every analysis is replicated
on the two larger datasets. These were run on a Kaggle T4 GPU (`scripts/09_kaggle_package.py`, about 4–6 h per
dataset); CMBan runs on a CPU.

## Tests

`python -m pytest -q tests` (27 tests) checks transliteration, determinism and monotone severity of the
perturbations, the metrics, that the CV engine keeps the modalities isolated, that augmentation only copies training
rows (no leakage into validation or test folds), and the OCR-cache fallback.

## Results

Full tables: [results/RESULTS.md](results/RESULTS.md). Primary pair XLM-R-base + CLIP ViT-B/32; macro-F1 %.

| | CMBan (133) | BanglaAbuseMeme (4,043) | MemoSen (4,368) |
|---|---|---|---|
| Best text encoder | 46.9 (mE5) | 60.5 (XLM-R-large) | 48.4 (XLM-R-large) |
| Char TF-IDF baseline | 46.2 | 65.2 | 52.6 |
| Best image encoder | 51.0 (CLIP-L/14) | 66.0 (CLIP-L/14) | 54.4 (CLIP-L/14) |
| XLM-R + CLIP-B/32 (best head) | 48.4 | 65.7 | 53.6 |
| Best fusion overall | 50.1 | 67.6 | 56.2 |
| Fusion > best text (Holm-significant tasks) | 0/5 | 4/4 | 1/1 |
| Fusion > best image (Holm-significant tasks) | 0/5 | 1/4 | 1/1 |
| Fusion drop, Romanised spelling noise (rel. %) | 20.7 | 11.5 | 7.7 |
| Fusion drop, blur (rel. %) | 2.6 | 6.3 | 7.4 |
| Fusion drop under noise, with translit. augmentation | 21 → 5 | 12 → 3 | 8 → 4 |

1. **Image > text.** CLIP-L/14 is the best single encoder everywhere, and a character n-gram TF-IDF baseline
   beats every frozen text encoder on BanglaAbuseMeme and MemoSen (MemoSen uses human captions, so this is not
   just OCR noise).
2. **Fusion helps with enough data.** Fusion is significantly better than text-only on all 5 tasks of the two
   large datasets and adds 1.5–2 points over image-only; on the 133 CMBan memes no difference is significant.
3. **Romanisation is the largest failure.** Writing the meme in Romanised Bangla is the most damaging perturbation
   on every dataset; native-script spelling noise and code-mixing are nearly harmless (≤ 3%). Fusion halves the
   damage on the large datasets but not on CMBan.
4. **Visual perturbations flip predictions without lowering F1 much** (up to 27% of predictions change); blur and
   down-sampling matter on the large datasets.
5. **Augmentation fixes most of it.** Adding transliterated copies of the training memes removes most of the
   Romanisation drop, including for the spelling-noise style never seen in training, at ≤ 0.7 F1 clean cost.
6. **Explainability.** Romanised text *gains* IG attribution share (0.55–0.59 → 0.67–0.73) instead of being
   discounted; visual degradation lowers it. Grad-CAM over-attends to rendered text only on cartoon memes (CMBan
   ratio 1.13 vs 0.76/0.88).
7. **Errors.** The largest error class is contextual/cultural (18% of CMBan, 10–11% elsewhere); cross-modal fusion
   errors are rare (1–4%).
