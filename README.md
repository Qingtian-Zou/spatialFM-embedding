# spatialFMs

A unified framework for extracting cell embeddings from spatial omics foundation models.

## Supported Models

| Model | Type | Embedding Dim | Status |
|---|---|---|---|
| **scGPT-spatial** | Transformer + MoE | 512 | Implemented |
| **Nicheformer** | Transformer MLM | 512 | Implemented |
| **Loki** (text / image) | Vision-language COCA ViT-L-14 | 768 | Implemented |

## Setup

```bash
# Install dependencies
pip install torch>=2.5.1 numpy anndata>=0.10 scanpy scipy scikit-learn einops numba>=0.59.0
```

### Model Weights

All weights are stored under `model_weights/` (gitignored) and must be obtained separately.

**scGPT-spatial** — Download the [scGPT-spatial weights](https://github.com/bowang-lab/scGPT-spatial?tab=readme-ov-file#-model-weights-) and place in `model_weights/scgpt_spatial/`:

```
model_weights/scgpt_spatial/
  best_model.pt          # 220 MB checkpoint
  vocab.json             # ~60,700 gene vocabulary
  args.json              # model hyperparameters
  all_dict_mean_std.csv  # normalization statistics
```

**Nicheformer** — Download the [Nicheformer checkpoint](https://github.com/theislab/nicheformer) and run the one-time conversion script:

```bash
# Requires pytorch-lightning (temporary install)
pip install pytorch-lightning
python scripts/convert_nicheformer_ckpt.py
pip uninstall pytorch-lightning -y
```

This produces `model_weights/nicheformer/`:

```
model_weights/nicheformer/
  model_state_dict.pt       # 197 MB pure-PyTorch state dict
  hparams.json              # model hyperparameters
  model.h5ad                # gene vocabulary (20,310 Ensembl IDs)
  {cosmx,dissociated,iss,merfish,xenium}_mean_script.npy  # platform-specific normalization
```

## Usage

```bash
# scGPT-spatial
python src/embed.py \
  --model scgpt_spatial \
  --input data.h5ad \
  --output output/ \
  --model-dir model_weights/scgpt_spatial/

# Nicheformer (input must use Ensembl gene IDs)
python src/embed.py \
  --model nicheformer \
  --input data.h5ad \
  --output output/ \
  --model-dir model_weights/nicheformer/ \
  --technology merfish

# Loki (text-only; add --spatial-dir for image embeddings)
python src/embed.py \
  --model loki \
  --input data.h5ad \
  --output output/ \
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

#### Nicheformer-specific Options

| Flag | Default | Description |
|---|---|---|
| `--technology` | `dissociated` | Platform for normalization: `cosmx`, `dissociated`, `iss`, `merfish`, or `xenium` |

#### Loki-specific Options

| Flag | Default | Description |
|---|---|---|
| `--spatial-dir` | *(none)* | Path to Visium `spatial/` folder (see below); enables image embedding |
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

## Architecture

The project uses an **adapter pattern**: each model has a wrapper in `src/adapters/` exposing a `run()` function. The CLI dispatches to the appropriate adapter based on `--model`.

```
src/
  embed.py              # CLI entry point
  adapters/
    scgpt_spatial.py    # scGPT-spatial adapter
    nicheformer.py      # Nicheformer adapter
    loki.py             # Loki adapter (text + image paths)
  models/
    scgpt_spatial/      # model code (with patches for torchtext and flash_attn compatibility)
    nicheformer/        # Nicheformer model code (pure nn.Module, ported from Lightning)
    loki/               # Loki model code (COCA ViT-L-14)
```

### scGPT-spatial Pipeline

Input `.h5ad` &rarr; slide-level mean normalization &rarr; per-gene population z-score &rarr; 51-bin quantile binning &rarr; tokenization via `GeneVocab` &rarr; transformer inference &rarr; 512-dim cell embeddings stored in `adata.obsm["X_scgpt_spatial"]`.

### Nicheformer Pipeline

Input `.h5ad` (Ensembl gene IDs) &rarr; gene alignment to 20,310-gene vocabulary &rarr; library-size normalization (10k) &rarr; platform-specific median normalization &rarr; rank tokenization (top genes by expression) &rarr; 12-layer transformer inference &rarr; mean-pooled 512-dim cell embeddings stored in `adata.obsm["X_nicheformer"]`.

### Loki Pipeline

**Text path:** Input `.h5ad` &rarr; top-50 genes per cell ranked by expression &rarr; gene-name sentence &rarr; COCA text encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_text"]`.

**Image path** (enabled via `--spatial-dir`): H&E image &rarr; per-cell patch extraction using spatial coordinates &rarr; COCA image encoder &rarr; 768-dim embeddings stored in `adata.obsm["X_loki_image"]`.

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_scgpt_spatial_tokenizer.py -v
```

Most tests use lightweight synthetic fixtures and run without model weights or sample data. Tests that require external files are skipped automatically via `requires_model_weights`, `requires_nicheformer_weights`, `requires_loki_weights`, and `requires_sample_data` markers.
