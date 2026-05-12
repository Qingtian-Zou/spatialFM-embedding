# spatialFMs

A unified framework for extracting cell embeddings from spatial omics foundation models.

## Supported Models

| Model | Type | Embedding Dim | Status |
|---|---|---|---|
| [**scGPT-spatial**](https://github.com/bowang-lab/scGPT-spatial) | Transformer + MoE | 512 | Implemented |
| [**Nicheformer**](https://github.com/theislab/nicheformer) | Transformer MLM | 512 | Planned |
| [**Loki** (text / image)](https://github.com/GuangyuWangLab2021/Loki) | Vision-language COCA ViT-L-14 | 768 | Implemented |
| [**STPath**](https://huggingface.co/tlhuang/STPath) | Spatial Transformer (generative) | 512 | Implemented |

## Setup

```bash
# Install dependencies
pip install -r requirements.txt
```

### Model Weights

Download model weights:
 - `model_weights/scgpt_spatial/`: [scGPT-spatial weights](https://github.com/bowang-lab/scGPT-spatial?tab=readme-ov-file#-model-weights-)
 - `model_weights/loki/`: [Loki weights](https://github.com/GuangyuWangLab2021/Loki?tab=readme-ov-file#pretrained-weights) &rarr; [checkpoint.pt](https://huggingface.co/WangGuangyuLab/Loki/blob/main/checkpoint.pt)
 - `model_weights/stpath/`:
   - [STPath weights](https://huggingface.co/tlhuang/STPath) &rarr; [stfm.pth](https://huggingface.co/tlhuang/STPath/blob/main/stfm.pth)
   - [symbol2ensembl.json](https://github.com/Graph-and-Geometric-Learning/STPath/blob/main/utils_data/symbol2ensembl.json)

```
model_weights/
  scgpt_spatial/
    best_model.pt        # 220 MB checkpoint
    vocab.json           # ~60,700 gene vocabulary
    args.json            # model hyperparameters
    all_dict_mean_std.csv  # normalization statistics
  loki/
    checkpoint.pt        # 7.2 GB checkpoint
  stpath/
    stfm.pth             # ~190 MB STFM state dict (filename as published on HF)
    symbol2ensembl.json  # 38,984 HGNC symbol -> Ensembl ID mapping
```

These files are gitignored and must be obtained separately.

#### STPath: Gigapath features (computed inline)

STPath consumes 1536-d tile features from [prov-gigapath](https://github.com/prov-gigapath/prov-gigapath), a ~1.5B-parameter H&E tile encoder. **Gigapath is loaded from a gated HuggingFace repo** — accept the EULA at [prov-gigapath/prov-gigapath](https://huggingface.co/prov-gigapath/prov-gigapath) and run `huggingface-cli login` (or set `HF_TOKEN`).

The adapter computes Gigapath features inline by default and caches them to `<output>/gigapath_features.h5`. Subsequent runs against the same output directory reuse the cache (no recomputation). The cache hit is decided by file existence alone — no automatic invalidation on input or parameter changes — so pass `--gigapath-recompute` to force re-encoding and overwrite a stale sidecar. You can:

- Provide a Visium spatial folder via `--spatial-dir` (uses `tissue_fullres_image.*` if present, else falls back to `tissue_hires_image.png` with a quality warning).
- Or pass an explicit `--fullres-image PATH` (strongly preferred — Gigapath was trained on ~0.5 mpp pathology tiles).
- Or rely on in-h5ad spatial metadata (`adata.uns['spatial'][lib]['images']`).
- Or short-circuit by supplying `--gigapath-h5 PATH` to a precomputed sidecar.

The sidecar (whether produced by the adapter or supplied externally) follows this schema:

| Dataset | Shape | dtype | Required |
|---|---|---|---|
| `embeddings` | `[n_spots, 1536]` | float32 | yes |
| `barcodes` | `[n_spots]` | string | yes |
| `coords` | `[n_spots, 2]` | float | optional (`adata.obsm['spatial']` takes precedence) |

The adapter aligns the gigapath sidecar to the AnnData by barcode intersection.

**When the sidecar is safe to reuse.** The cached embeddings are a function of *the per-spot pixel crops fed to the encoder*. Anything that changes those crops invalidates the cache; anything that doesn't is free to vary between runs. The adapter never checks this, so please pass `--gigapath-recompute` (or delete the sidecar) when an invalidating input changes.

Requires recomputation (`--gigapath-recompute`):

| Change | Why it invalidates |
|---|---|
| Different H&E image (`--fullres-image`, `--spatial-dir`, or `adata.uns['spatial'][lib]['images']`) | Crops are sampled from a different image. |
| Different `adata.obsm['spatial']` coordinates for the same barcodes | Crops are centered on different pixels. |
| Different `--patch-px`, or different `spot_diameter_fullres` in the Visium `scalefactors_json.json` | Crop side length changes. |
| Switching from fullres to hires (or vice versa) for the same `--spatial-dir` | Image pixel space and the auto-scaled `patch_px` both change. |
| Different `--library-id` when `adata.uns['spatial']` has multiple entries | A different image entry may be selected. |

Safe to change without recomputation (these don't affect the cached values):

- `--gigapath-batch-size` — throughput only.
- `--gigapath-precision` (`fp32` ↔ `fp16`) — note: a cached sidecar keeps the precision it was originally written with; toggling the flag will *not* trigger a re-encode at the new precision.
- `--device` (`cuda` ↔ `cpu`) — same outputs modulo float rounding.
- STPath-side knobs that never touch Gigapath: `--organ-type`, `--tech-type`, `--save-imputed-expression`, the STPath weights in `--model-dir`.

**Subset / superset of barcodes.** Alignment is by barcode intersection ([src/adapters/stpath.py:305](src/adapters/stpath.py#L305)). A cached sidecar with *more* barcodes than the current AnnData is fine — extras are ignored. A cached sidecar with *fewer* barcodes than the current AnnData will silently subset the AnnData down to the intersection (a one-line `[stpath] Barcode alignment: N/M` log is the only signal). If you've *added* spots since the cache was written, please recompute by passing `--gigapath-recompute`.

## Usage

```bash
# scGPT-spatial
python src/embed.py \
  --model scgpt_spatial \
  --input data.h5ad \
  --output output/scgpt_spatial/ \
  --model-dir model_weights/scgpt_spatial/

# Loki — produces text embeddings always, plus image embeddings when the
# input .h5ad carries spatial metadata. Use --spatial-dir to override the
# h5ad with a Visium spatial/ folder.
python src/embed.py \
  --model loki \
  --input data.h5ad \
  --output output/loki/ \
  --model-dir model_weights/loki/

# STPath — full-context imputation forward pass produces a unified
# image+expression+coords joint embedding. Gigapath tile features are
# computed inline by default and cached to <output>/gigapath_features.h5;
# pass --gigapath-h5 to reuse a precomputed sidecar.
python src/embed.py \
  --model stpath \
  --input data.h5ad \
  --output output/stpath/ \
  --model-dir model_weights/stpath/ \
  --spatial-dir spatial/ \
  --organ-type Kidney --tech-type Visium
```

### CLI Options

#### General Options

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | `scgpt_spatial`, `nicheformer`, `loki`, or `stpath` |
| `--input` | *(required)* | Path to input `.h5ad` file |
| `--output` | *(required)* | Output directory for embedding files |
| `--model-dir` | *(required)* | Path to model weights directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--gene-col` | `feature_name` | Column in `adata.var` for gene names, or `index` |
| `--max-length` | `1200` | Maximum sequence length |
| `--batch-size` | `64` | Batch size for inference |

#### Loki-specific Options

| Flag | Default | Description |
|---|---|---|
| `--spatial-dir` | *(none)* | (loki, stpath) Path to Visium `spatial/` folder (see below). For loki: overrides spatial data in the input `.h5ad` (falls back to h5ad on read failure). For stpath: source folder for inline Gigapath feature extraction. |
| `--housekeeping-genes` | *(none)* | (loki) CSV with a `genesymbol` column; genes to exclude from text encoding |
| `--library-id` | *(auto)* | (loki, stpath) Key under `adata.uns['spatial']` (default: first found) |
| `--patch-size` | `16` | (loki) H&E patch side length in pixels |

#### STPath-specific Options

| Flag | Default | Description |
|---|---|---|
| `--gigapath-h5` | *(none)* | Optional precomputed Gigapath sidecar `.h5`. When omitted, features are computed inline and cached at `<output>/gigapath_features.h5` |
| `--fullres-image` | *(none)* | Full-resolution H&E image (preferred over the Visium hires PNG for inline Gigapath encoding) |
| `--patch-px` | *(auto)* | Per-spot crop side length in image pixels. Defaults to `spot_diameter_fullres` from Visium scalefactors (auto-scaled for hires) |
| `--gigapath-batch-size` | `32` | Batch size for inline Gigapath inference |
| `--gigapath-precision` | `fp32` | `fp32` or `fp16` for inline Gigapath inference |
| `--gigapath-cache` | *(auto)* | Sidecar cache path written/read when `--gigapath-h5` is not supplied (default: `<output>/gigapath_features.h5`) |
| `--gigapath-recompute` | off | Force re-computation of the Gigapath sidecar `.h5` even when a cache exists at `--gigapath-cache`; overwrites in place. Mutually exclusive with `--gigapath-h5`. |
| `--organ-type` | `Others` | One of 25 STPath organ tokens (e.g. `Kidney`, `Brain`, `Lung`, `Breast`, `Liver`, `Heart`, ...; see `src/models/stpath/utils/constants.py:organ_voc`) |
| `--tech-type` | *(pad)* | One of `Spatial Transcriptomics`, `Visium`, `Xenium`, `Visium HD` |
| `--save-imputed-expression` | off | Also write `imputed_expression.h5ad` with the model's refined log1p expression on STPath's 38,984-gene vocabulary |

The `--spatial-dir` folder should be a standard Visium `spatial/` directory containing:

```
spatial/
  scalefactors_json.json   # scale factors (including tissue_hires_scalef)
  tissue_hires_image.png   # high-resolution H&E image
  tissue_positions.csv     # barcode-to-pixel coordinate mapping
```

### Output Formats

Each run produces four output files: `.h5ad`, `.npy`, `.csv`, and `.tsv`.

Loki processes expression data and spatial context separately. The text (expression) embedding is always produced; the image (spatial) embedding is produced whenever spatial data is available — either embedded in the input `.h5ad` (`obsm['spatial']` + `uns['spatial']`) or supplied via `--spatial-dir`. When `--spatial-dir` is set, it *overrides* the in-h5ad spatial data; if the folder cannot be read, the adapter prints a warning and falls back to whatever spatial data the h5ad carries (or text-only if none). When both modalities are produced, there are 2 sets of output files except for `.h5ad`. For `.h5ad`, expression embeddings are saved as `adata.obsm["X_loki_text"]` and spatial embeddings as `adata.obsm["X_loki_image"]`, so there is only one `.h5ad` file.

## Architecture

The project uses an **adapter pattern**: each model has a wrapper in `src/adapters/` exposing a `run()` function. The CLI dispatches to the appropriate adapter based on `--model`.

```
src/
  embed.py              # CLI entry point
  adapters/
    scgpt_spatial.py    # scGPT-spatial adapter
    loki.py             # Loki adapter (text + image paths)
    stpath.py           # STPath adapter (full-context joint embedding)
  models/
    scgpt_spatial/      # model code (with patches for torchtext and flash_attn compatibility)
    loki/               # Loki model code (COCA ViT-L-14)
    stpath/             # STPath STFM (spatial transformer + multi-modal input encoder)
```

### scGPT-spatial Pipeline

Input `.h5ad` &rarr; slide-level mean normalization &rarr; per-gene population z-score &rarr; 51-bin quantile binning &rarr; tokenization via `GeneVocab` &rarr; transformer inference &rarr; 512-dim cell embeddings stored in `adata.obsm["X_scgpt_spatial"]`.

### Loki Pipeline

**Text path:** Input `.h5ad` &rarr; top-50 genes per cell ranked by expression &rarr; gene-name sentence &rarr; COCA text encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_text"]`.

**Image path** (produced when spatial data is available — either inside the input `.h5ad` or supplied via `--spatial-dir`): H&E image &rarr; per-cell patch extraction using spatial coordinates &rarr; COCA image encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_image"]`.

The saved per-spot patches are 3-channel RGB **uint8** PNGs because OmiCLIP's image transform (`Resize → CenterCrop → ToTensor → OpenAI-CLIP Normalize`) is calibrated for that format — `ToTensor` divides uint8 PIL pixels by 255 internally. The adapter's `_resolve_spatial` normalizes the hires image to uint8 RGB at load time, so both input routes (Visium PNG via `PIL.Image.open` and the in-memory cache from `adata.uns['spatial']`) produce equivalent patches regardless of the source dtype (uint8, float `[0, 1]`, uint16) or channel count (RGB or RGBA).

### STPath Pipeline

Input `.h5ad` (with `obsm['spatial']`, raw counts, HGNC gene symbols) + Gigapath sidecar `.h5` (1536-d H&E features, barcodes) &rarr; barcode alignment &rarr; coord rescale to `[0, 100]` &rarr; `log1p` of raw counts &rarr; symbol→Ensembl→token-id mapping (drops genes outside STPath's 38,984-vocab; refuses to run if fewer than 100 mapped) &rarr; multi-hot expression tensor on STPath's vocabulary &rarr; STFM full-context forward pass (`prediction_head(..., return_all=True)`) &rarr; 512-dim hidden state stored in `adata.obsm["X_stpath"]`. With `--save-imputed-expression`, the model's refined log1p expression is written as a sibling `imputed_expression.h5ad` (shape `(n_obs, 38984)`).

**Caveats:**

- `X_stpath` is a **multi-modal joint embedding** (H&E morphology + transcriptome + spatial coordinates + organ/tech). It is NOT a cell-state embedding and is NOT directly comparable with `X_scgpt_spatial` / `X_loki_text` / `X_nicheformer` (purely transcriptomic) or `X_loki_image` (purely visual).
- The all-context forward pass is **out-of-distribution** vs. STPath's training mask ratios in `[0.1, 0.95]`. Validate on a downstream task (clustering, niche recovery, etc.) before relying on it.
- **Human-only** — STPath's vocabulary is 38,984 human Ensembl IDs. Mouse data needs an orthology mapping step that this adapter does not provide.
- Inputs must use **HGNC gene symbols** in `var_names`. Inputs that use raw Ensembl IDs need an upstream conversion step.

### Patches Applied to Upstream Code

Where our vendored copies diverge from the upstream references:

- **Loki — `src/models/loki/preprocess.py::segment_patches`**: changed the per-spot coordinate unpacking from `ycenter, xcenter = coord[..., ["pixel_x", "pixel_y"]]` to `xcenter, ycenter = ...`. The OmiCLIP weights themselves are fine — upstream's image pipeline is internally self-consistent: its `load_data_for_annotation` stores rows in `pixel_x` and cols in `pixel_y` (swap #1), and `segment_patches` reads them with a matching swap (swap #2), so the two cancel and the model was trained on correctly aligned patches. Our adapter [src/adapters/loki.py](src/adapters/loki.py) does not use upstream's `load_data_for_annotation`; it builds the coord DataFrame with intuitive naming (`pixel_x` = x-axis = col, `pixel_y` = y-axis = row), which removes swap #1. This patch removes swap #2 to match, so inference patches land on the same pixels the model saw at training time.

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_scgpt_spatial_tokenizer.py -v
```

Most tests use lightweight synthetic fixtures and run without model weights or sample data. Tests that require external files are skipped automatically via `requires_model_weights` and `requires_sample_data` markers.
