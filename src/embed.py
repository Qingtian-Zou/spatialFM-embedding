"""Unified CLI entry point for spatial foundation model embeddings."""

import argparse
import os
import sys

# Ensure the project root is on sys.path for direct script execution
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def main():
    parser = argparse.ArgumentParser(
        description="Generate cell embeddings from spatial foundation models.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["scgpt_spatial", "nicheformer", "loki_text", "loki_image"],
        help="Model to use for embedding.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input .h5ad file.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for embedding files.",
    )
    parser.add_argument(
        "--model-dir",
        required=True,
        help="Path to model weights directory.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cuda).",
    )
    parser.add_argument(
        "--gene-col",
        default="feature_name",
        help="Column in adata.var for gene names, or 'index' (default: feature_name).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=1200,
        help="Maximum sequence length (default: 1200).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64).",
    )

    args = parser.parse_args()

    if args.model == "scgpt_spatial":
        from src.adapters.scgpt_spatial import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            gene_col=args.gene_col,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.model in ("nicheformer", "loki_text", "loki_image"):
        print(f"Error: Model '{args.model}' is not yet implemented.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
