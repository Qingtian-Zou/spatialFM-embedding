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
        choices=["scgpt_spatial", "nicheformer", "loki"],
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
    parser.add_argument(
        "--spatial-dir",
        default=None,
        help="(loki) Path to a Visium spatial/ folder. When set, overrides spatial data in the input .h5ad; falls back to h5ad on read failure.",
    )
    parser.add_argument(
        "--housekeeping-genes",
        default=None,
        help="(loki) CSV with a 'genesymbol' column; genes to exclude from text encoding.",
    )
    parser.add_argument(
        "--library-id",
        default=None,
        help="(loki) Key under adata.uns['spatial'] (default: first).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=16,
        help="(loki) H&E patch side length in pixels (default: 16).",
    )
    parser.add_argument(
        "--technology",
        default="dissociated",
        choices=["cosmx", "dissociated", "iss", "merfish", "xenium"],
        help="(nicheformer) Platform/technology for normalization (default: dissociated).",
    )
    parser.add_argument(
        "--no-symbol-conversion",
        dest="convert_symbols",
        action="store_false",
        default=True,
        help="(nicheformer) Disable automatic HGNC symbol -> Ensembl ID conversion.",
    )
    parser.add_argument(
        "--hgnc-mapping",
        default=None,
        help="(nicheformer) Path to a custom HGNC TSV. Defaults to the bundled file.",
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
    elif args.model == "loki":
        from src.adapters.loki import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            spatial_dir=args.spatial_dir,
            housekeeping_genes_path=args.housekeeping_genes,
            library_id=args.library_id,
            patch_size=args.patch_size,
            device=args.device,
        )
    elif args.model == "nicheformer":
        from src.adapters.nicheformer import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            technology=args.technology,
            batch_size=args.batch_size,
            device=args.device,
            convert_symbols=args.convert_symbols,
            hgnc_mapping_path=args.hgnc_mapping,
        )


if __name__ == "__main__":
    main()
