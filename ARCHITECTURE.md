# ToonSense Architecture

This document describes how ToonSense is built: the data flow, the model architecture, the evaluation design and
the code layout. For how to run it, see [README.md](README.md).

## 1. System overview

```
                         ┌──────────── Meme: CMBan · BanglaAbuseMeme · MemoSen (image with embedded text) ───────────┐
                         │                                                                                                │
                         ▼                                                                                                ▼
              ┌─────────────────────┐                                                                      ┌──────────────────────┐
              │  OCR (EasyOCR bn+en)│──► text boxes (reused for text-region occlusion)                     │  Letterbox to square │
              └──────────┬──────────┘                                                                      │  (median border col.)│
                         │ OCR text (or gold caption / manual transcript)                                  └──────────┬───────────┘
                         ▼                                                                                            ▼
              ┌─────────────────────┐                                                                      ┌──────────────────────┐
              │ Linguistic          │  translit · roman_noise · bangla_noise                               │ Visual perturbation  │
              │ perturbation (opt.) │  codemix · emoji_add · emoji_remove                                  │ (optional)           │ blur · crop · resize · jpeg
              └──────────┬──────────┘                                                                      └──────────┬───────────┘ text_occlusion · region_occlusion
                         ▼                                                                                            ▼
              ┌─────────────────────┐                                                                      ┌──────────────────────┐
              │ Frozen text encoder │  XLM-R base (primary), XLM-R large, mBERT,                          │ Frozen image encoder │  CLIP ViT-B/32 (primary),
              │  → pooled vector t  │  BanglaBERT, LaBSE, mE5                                              │  → vector v          │  CLIP B/16, L/14, DINOv2
              └──────────┬──────────┘                                                                      └──────────┬───────────┘
                         │                         cached to features/<ds>/text/<enc>/<variant>.npz                      │  cached to features/<ds>/image/<enc>/<variant>.npz
                         └───────────────────────────────────────┬──────────────────────────────────────────────────┘
                                                                 ▼
                                                  ┌───────────────────────────────┐
                                                  │ Standardise (fit on train fold)│
                                                  └───────────────┬───────────────┘
                                                                  ▼
                                                  ┌───────────────────────────────┐
                                                  │ Classification head            │  unimodal (LR / MLP) or
                                                  │ (one per task)                 │  fusion (concat LR / concat MLP / GMU / bilinear / late)
                                                  └───────────────┬───────────────┘
                                                                  ▼
                     CMBan: humour · sarcasm · offensive · motivational · sentiment
                     BanglaAbuseMeme: abuse · vulgarity · sarcasm · sentiment   |   MemoSen: sentiment
                                                                  │
          ┌──────────────────────────────┬────────────────────────┼────────────────────────┬──────────────────────────────┐
          ▼                              ▼                        ▼                        ▼                              ▼
  Metrics + significance     Robustness Drop + flip rate   Explainability (IG,     Error categorisation        Mitigation: Romanisation
                                                           Grad-CAM)                                           augmentation (heads only)
```

Design principle: **the encoders are frozen and their embeddings are pre-computed**, as the proposal specifies.
Only the small heads are trained. A perturbation only requires re-encoding its inputs, not retraining anything, and
the augmentation experiment reuses the cached embeddings without any encoder pass. CMBan (133 memes) runs on a CPU;
BanglaAbuseMeme and MemoSen (~4k memes each) run on a Kaggle GPU session (`scripts/09_kaggle_package.py`).

## 2. Input processing

### 2.1 Text modality
CMBan and BanglaAbuseMeme memes have their text rendered inside the image, so text comes from **EasyOCR** with
Bangla and English models (`src/toonsense/ocr.py`). MemoSen ships human-written captions, which are used as the
text (`text_column` in the dataset config); OCR still runs on it for text boxes and confidences.

- Very large images are downscaled so their longest side is at most 1280 px.
- Fragments are sorted into reading order: top-to-bottom line bands, then left-to-right.
- Fragments with confidence below 0.10 are dropped.
- Output is cached per image as JSON: text, bounding box and confidence.
- Text priority: `data/transcripts_manual_<dataset>.csv` (if present) > dataset-provided caption > OCR.
- If the per-image JSON cache is missing (e.g. only `texts.csv` was copied back from a Kaggle run), the summary
  file `data/ocr/<dataset>/texts.csv` is used instead.

### 2.2 Image modality
CLIP and DINOv2 processors normally **centre-crop** to a square, which cuts the top and bottom panels off tall
multi-panel comics. Each image is therefore **letterboxed** to a square using its median border colour before
preprocessing (`pad_to_square` in `encoders.py`), so the encoder sees the whole meme at 224×224.

## 3. Encoders (`src/toonsense/encoders.py`)

| Modality | Name | Hugging Face ID | Output | Pooling |
|---|---|---|---|---|
| Text | **xlmr-base** (primary) | FacebookAI/xlm-roberta-base | 768 | mean over tokens |
| Text | xlmr-large | FacebookAI/xlm-roberta-large | 1024 | mean |
| Text | mbert | google-bert/bert-base-multilingual-cased | 768 | mean |
| Text | banglabert | csebuetnlp/banglabert | 768 | mean |
| Text | labse | sentence-transformers/LaBSE | 768 | pooler, L2-normalised |
| Text | me5-base | intfloat/multilingual-e5-base | 768 | mean, `"query: "` prefix, L2-normalised |
| Image | **clip-b32** (primary) | openai/clip-vit-base-patch32 | 512 | CLIP projected image embedding |
| Image | clip-b16 | openai/clip-vit-base-patch16 | 512 | CLIP projected image embedding |
| Image | clip-l14 | openai/clip-vit-large-patch14 | 768 | CLIP projected image embedding |
| Image | dinov2-base | facebook/dinov2-base | 1536 | [CLS ; mean of patch tokens] |

Text is truncated to 256 tokens. Embeddings are cached per (modality, encoder, variant), where the variant is
`clean` or a perturbation such as `translit_s3`.

## 4. Classification heads (`src/toonsense/heads.py`)

Every head standardises its inputs (z-score fitted on the training fold) and is trained separately for each task.
Notation: `t` is the text vector, `v` the image vector, H = 256 the hidden size, p = 0.3 the dropout, C the
number of classes.

| Head | Type | Architecture |
|---|---|---|
| `logreg` | unimodal | L2 logistic regression on `t` or `v`. C chosen from {0.01, 0.1, 1} on the validation split; balanced class weights |
| `mlp` | unimodal | Dropout(p/2) → Linear(d, H) → GELU → Dropout(p) → Linear(H, C) |
| `concat_lr` | early fusion | Logistic regression on `[t ; v]` |
| `concat_mlp` | early fusion | The MLP above applied to `[t ; v]` |
| `gmu` | gated fusion | Gated Multimodal Unit (Arevalo et al., 2017): `h_t = tanh(W_t t)`, `h_v = tanh(W_v v)`, `z = σ(W_z [t ; v])`, `h = z ⊙ h_t + (1 − z) ⊙ h_v` → Dropout → Linear(H, C). The gate `z` is logged as a text-reliance signal |
| `bilinear` | interaction fusion | `a = tanh(P_t t)`, `b = tanh(P_v v)` → `[a ; b ; a⊙b ; |a − b|]` → Linear(4H, H) → GELU → Linear(H, C) |
| `late` | decision fusion | Mean of the text-MLP and image-MLP class probabilities |

Training of the neural heads:
- Full-batch AdamW (lr 1e-3, weight decay 1e-2) with class-weighted cross-entropy.
- Up to 300 epochs, with early stopping (patience 30) on validation loss; the best checkpoint is restored.

Baselines: the majority class, and character n-gram (2–5) TF-IDF with logistic regression, which is
script-agnostic and so handles code-mixing.

## 5. Tasks and labels (`src/toonsense/data.py`)

| Task | Fine labels (original CMBan) | Coarse labels (collapsed) |
|---|---|---|
| Humour | Not Funny / Funny / Very Funny | Not Funny / Funny |
| Sarcasm | Not / Little / Very Sarcastic | Not Sarcastic / Sarcastic |
| Offensive | Not / Slight / Hateful Offensive | Not Offensive / Offensive |
| Motivational | Not Motivational / Motivational | (same) |
| Sentiment | Very Negative … Very Positive (5) | Negative / Neutral / Positive |

The language tags are grouped into Bangla, Code-mixed and English, for analysis broken down by language.

| Dataset | Memes | Tasks (classes) | Text |
|---|---|---|---|
| CMBan (released subset) | 133 | the five tasks above (primary level: coarse) | OCR |
| BanglaAbuseMeme | 4,043 | sentiment (3), sarcasm (2), abuse (2), vulgarity (2) | OCR |
| MemoSen | 4,368 | sentiment (3) | human captions |

The two larger datasets carry no language tags, so each meme gets a script group from its text: Bangla (≥ 85% of
letters in Bangla script), Latin-script (≤ 15%) or Code-mixed. Datasets are selected with `--dataset` (or
`$TOONSENSE_DATASET`); `config.activate()` refreshes the task registry in place.

## 6. Evaluation design (`src/toonsense/cv.py`, `metrics.py`)

```
for seed in {0, 1, 2}:
    StratifiedKFold(5) ──► for each fold:
        training part (80%) ──► stratified inner split ──► fit (85%) + validation (15%, early stopping / C)
        test part (20%)    ──► predict on clean AND on every perturbed copy of the same test memes
    ► out-of-fold (OOF) predictions: every meme predicted exactly once per seed
```

- **Clean metrics**: accuracy, macro precision, macro recall, macro-F1 and weighted F1 on the OOF predictions,
  reported as mean ± std over seeds.
- **Uncertainty**: 95% bootstrap CI (1,000 resamples) on the seed-ensembled predictions.
- **Significance**: paired bootstrap test of the best fusion model against the best text-only and best image-only
  models, with Holm correction across tasks.
- **Robustness**: the *same* trained fold models score the clean and perturbed versions of their test fold.
  - Robustness Drop: `RD_abs = F1_clean − F1_pert`, `RD_rel = 100 · RD_abs / F1_clean`.
  - Flip rate: the share of predictions that change. It doesn't need labels.
  - Every measure is also broken down by language group.
- **Settings**:
  - *Decoupled* (default): only one modality is perturbed at a time.
  - *Coupled* (optional): perturbed images are re-OCRed, so text degrades as it would in real use.

## 7. Perturbations

All perturbations are deterministic given (meme ID, family, severity), via a CRC32-seeded RNG.

**Linguistic** (`perturb_text.py`):

| Family | Mechanism | Severity 1 / 2 / 3 |
|---|---|---|
| translit | Rule-based Bangla → colloquial Banglish transliterator: inherent-vowel handling, schwa deletion in VC·CV context, ya- and ba-phala, nukta forms | 33 / 66 / 100% of Bangla words |
| roman_noise | Full romanisation, then Banglish spelling variants (sh↔s, bh→v, i→ee, vowel drop, letter repetition, swaps) | 15 / 30 / 50% of words |
| bangla_noise | Bangla orthographic confusions (ি/ী, ু/ূ, শ/ষ/স, ন/ণ, জ/য, র/ড়, …) and dropped vowel signs | 10 / 20 / 35% of words |
| codemix | Bangla↔English substitution from a ~200-word lexicon, with suffix stripping | p = 0.33 / 0.66 / 1.0 |
| emoji_add | Insert emojis from a mixed-sentiment pool | 1 / 3 / 6 emojis |
| emoji_remove | Remove emojis → also emoticons → also all symbols | three cumulative levels |

**Visual** (`perturb_image.py`):

| Family | Mechanism | Severity 1 / 2 / 3 |
|---|---|---|
| blur | Gaussian blur with σ defined at 224-px scale (resolution-independent) | σ = 1 / 2 / 4 |
| crop | Random crop | keeps 85 / 70 / 55% of the area |
| resize | Down-sample, then up-sample back | longest side 160 / 96 / 48 px |
| jpeg | JPEG re-compression | quality 30 / 15 / 5 |
| text_occlusion | Fill OCR text boxes with their median colour | 33 / 66 / 100% of boxes |
| region_occlusion | Grey cells on a 12×12 grid, restricted to non-text areas | 10 / 25 / 40% of the non-text area |

## 8. Explainability (`src/toonsense/explain.py`)

| Method | Target | What it measures |
|---|---|---|
| Modality IG | Fusion-head inputs `[t ; v]`, training-mean baseline, 32 midpoint steps | Text share of the attribution for every meme, clean vs perturbed |
| Token IG | Input embeddings of the text encoder, pad-token baseline, subwords merged into words | Which words drive a prediction, before vs after a linguistic perturbation |
| Grad-CAM / Grad-CAM++ | CLIP ViT patch tokens at the **input of the last block**. At the last block's output, only the CLS token feeds the projection, so patch gradients would be zero | (i) Share of attention on rendered text relative to its area; (ii) clean-vs-perturbed CAM correlation |

## 9. Error analysis (`scripts/06_error_analysis.py`)

Errors of the fusion model (the best *learned* fusion head; late fusion is excluded because, as an average of the
two unimodal models, it cannot fail on binary tasks when both are right) are labelled by comparing them with the
unimodal models trained on the same folds:

| Text-only | Image-only | Fusion | Category |
|---|---|---|---|
| ✗ | ✓ | ✗ | linguistic |
| ✓ | ✗ | ✗ | visual |
| ✓ | ✓ | ✗ | cross-modal (fusion failure) |
| ✗ | ✗ | ✗ | OCR-limited (empty or low-confidence text), else **contextual/cultural candidate**, to be confirmed manually |

Robustness fragility: a correct clean prediction becomes wrong at severity ≥ 2 under a linguistic perturbation
only, a visual one only, or both (cross-modal-fragile).

## 10. Mitigation: Romanisation augmentation (`scripts/11_augmentation.py`)

`run_cv(..., augment=[(Xt_aug, Xv_aug), ...])` appends, in every fold, a copy of each **non-validation training**
meme with its text embedding replaced by a perturbed variant (clean image embedding, same label). Validation and
test memes are never copied (checked by `test_augmentation_uses_only_training_rows`).

| Setting | Added training copies | Purpose |
|---|---|---|
| none | – | baseline (identical to step 04) |
| +translit. | `translit_s2` | Romanised spelling noise stays unseen, which tests generalisation |
| +translit. + noise | `translit_s2`, `roman_noise_s2` | upper bound |

Each (task, setting, seed) is one job, written to `results/<dataset>/augmentation/parts/` so that an interrupted
run resumes. Feature arrays are loaded once and shared with the workers as read-only memory maps.

## 11. Code layout

```
configs/default.yaml        all paths, encoders, heads, CV and perturbation settings
src/toonsense/
  config.py                 config loading, environment (HF cache on D:, UTF-8 console)
  data.py                   label loading, task taxonomies, language groups
  ocr.py                    EasyOCR wrapper + cache, manual transcript override
  perturb_text.py           linguistic perturbations + Bangla→Banglish transliterator
  perturb_image.py          visual perturbations
  encoders.py               frozen text/image encoders, letterboxing, feature cache
  heads.py                  LR / MLP / GMU / bilinear heads behind one fit/predict interface
  cv.py                     repeated stratified CV with OOF evaluation on clean + perturbed inputs, optional augmentation
  metrics.py                scores, bootstrap CI, paired bootstrap, Robustness Drop, flip rate
  explain.py                modality IG, token IG, Grad-CAM(++), text-attention ratio
scripts/01_ocr.py … 11_augmentation.py   pipeline steps, paper figures/tables, Kaggle packaging (see README)
tests/test_core.py          unit tests
results/<dataset>/          benchmark/, robustness/, augmentation/, explain/, error_analysis/, tables/, figures/
```
