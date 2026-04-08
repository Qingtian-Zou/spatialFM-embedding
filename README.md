# spatialFMs

A unified framework for extracting cell embeddings from spatial omics foundation models.

## Supported Models

| Model | Type | Embedding Dim | Status |
|---|---|---|---|
| **scGPT-spatial** | Transformer + MoE | 512 | Implemented |
| **Nicheformer** | Transformer MLM | 512 | Planned |
| **Loki** (text / image) | Vision-language COCA ViT-L-14 | 768 | Planned |

## Setup

```bash
# Install dependencies
pip install torch>=2.5.1 numpy anndata>=0.10 scanpy scipy scikit-learn einops
```

### Model Weights

Download the [scGPT-spatial weights](https://github.com/bowang-lab/scGPT-spatial?tab=readme-ov-file#-model-weights-) and place them in `model_weights/scgpt_spatial/`:

```
models/scgpt_spatial/
  best_model.pt        # 220 MB checkpoint
  vocab.json           # ~60,700 gene vocabulary
  args.json            # model hyperparameters
  all_dict_mean_std.csv  # normalization statistics
```

These files are gitignored and must be obtained separately.

## Usage

```bash
python src/embed.py \
  --model scgpt_spatial \
  --input data.h5ad \
  --output output/ \
  --model-dir model_weights/scgpt_spatial/ \
```

### CLI Options

| Flag | Default | Description |
|---|---|---|
| `--model` | *(required)* | `scgpt_spatial`, `nicheformer`, `loki_text`, or `loki_image` |
| `--input` | *(required)* | Path to input `.h5ad` file |
| `--output` | *(required)* | Output directory for embedding files |
| `--model-dir` | *(required)* | Path to model weights directory |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--gene-col` | `feature_name` | Column in `adata.var` for gene names, or `index` |
| `--max-length` | `1200` | Maximum sequence length |
| `--batch-size` | `64` | Batch size for inference |

### Output Formats

Each run produces four output files: `.h5ad`, `.npy`, `.csv`, and `.tsv`.

## Architecture

The project uses an **adapter pattern**: each model has a wrapper in `src/adapters/` exposing a `run()` function. The CLI dispatches to the appropriate adapter based on `--model`.

```
src/
  embed.py              # CLI entry point
  adapters/
    scgpt_spatial.py    # scGPT-spatial adapter
  models/
    scgpt_spatial/      # model code (with patches for torchtext and flash_attn compatibility)
```

### scGPT-spatial Pipeline

Input `.h5ad` &rarr; slide-level mean normalization &rarr; per-gene population z-score &rarr; 51-bin quantile binning &rarr; tokenization via `GeneVocab` &rarr; transformer inference &rarr; 512-dim cell embeddings stored in `adata.obsm["X_scgpt_spatial"]`.

## Testing

```bash
# Run all tests
pytest tests/

# Run a specific test file
pytest tests/test_scgpt_spatial_tokenizer.py -v
```

Most tests use lightweight synthetic fixtures and run without model weights or sample data. Tests that require external files are skipped automatically via `requires_model_weights` and `requires_sample_data` markers.
