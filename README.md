# spatialFMs

A unified framework for extracting cell embeddings from spatial omics foundation models.

## Supported Models

| Model | Type | Embedding Dim | Status |
|---|---|---|---|
| [**scGPT-spatial**](https://github.com/bowang-lab/scGPT-spatial) | Transformer + MoE | 512 | Implemented |
| [**Nicheformer**](https://github.com/theislab/nicheformer) | Transformer MLM | 512 | Planned |
| [**Loki** (text / image)](https://github.com/GuangyuWangLab2021/Loki) | Vision-language COCA ViT-L-14 | 768 | Implemented |

## Setup

```bash
# Install dependencies
pip install torch>=2.5.1 numpy anndata>=0.10 scanpy scipy scikit-learn einops
```

### Model Weights

Download model weights:
 - `model_weights/scgpt_spatial/`: [scGPT-spatial weights](https://github.com/bowang-lab/scGPT-spatial?tab=readme-ov-file#-model-weights-)
 - `model_weights/loki/`: [Loki weights](https://github.com/GuangyuWangLab2021/Loki?tab=readme-ov-file#pretrained-weights) &rarr; [checkpoint.pt](https://huggingface.co/WangGuangyuLab/Loki/blob/main/checkpoint.pt)

```
models/
  scgpt_spatial/
    best_model.pt        # 220 MB checkpoint
    vocab.json           # ~60,700 gene vocabulary
    args.json            # model hyperparameters
    all_dict_mean_std.csv  # normalization statistics
  loki/
    checkpoint.pt        # 7.2 GB checkpoint
```

These files are gitignored and must be obtained separately.

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
```

### CLI Options

#### General Options

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | `scgpt_spatial`, `nicheformer`, or `loki` |
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
| `--spatial-dir` | *(none)* | Path to Visium `spatial/` folder (see below). When set, overrides spatial data in the input `.h5ad`; falls back to h5ad on read failure |
| `--housekeeping-genes` | *(none)* | CSV with a `genesymbol` column; genes to exclude from text encoding |
| `--library-id` | *(auto)* | Key under `adata.uns['spatial']` (default: first found) |
| `--patch-size` | `16` | H&E patch side length in pixels |

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
  models/
    scgpt_spatial/      # model code (with patches for torchtext and flash_attn compatibility)
    loki/               # Loki model code (COCA ViT-L-14)
```

### scGPT-spatial Pipeline

Input `.h5ad` &rarr; slide-level mean normalization &rarr; per-gene population z-score &rarr; 51-bin quantile binning &rarr; tokenization via `GeneVocab` &rarr; transformer inference &rarr; 512-dim cell embeddings stored in `adata.obsm["X_scgpt_spatial"]`.

### Loki Pipeline

**Text path:** Input `.h5ad` &rarr; top-50 genes per cell ranked by expression &rarr; gene-name sentence &rarr; COCA text encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_text"]`.

**Image path** (produced when spatial data is available — either inside the input `.h5ad` or supplied via `--spatial-dir`): H&E image &rarr; per-cell patch extraction using spatial coordinates &rarr; COCA image encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_image"]`.

The saved per-spot patches are 3-channel RGB **uint8** PNGs because OmiCLIP's image transform (`Resize → CenterCrop → ToTensor → OpenAI-CLIP Normalize`) is calibrated for that format — `ToTensor` divides uint8 PIL pixels by 255 internally. The adapter's `_resolve_spatial` normalizes the hires image to uint8 RGB at load time, so both input routes (Visium PNG via `PIL.Image.open` and the in-memory cache from `adata.uns['spatial']`) produce equivalent patches regardless of the source dtype (uint8, float `[0, 1]`, uint16) or channel count (RGB or RGBA).

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
