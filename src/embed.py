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
        choices=["scgpt_spatial", "nicheformer", "loki", "stpath"],
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
        default=None,
        help="Batch size for inference (default: 64). Ignored for --model stpath "
             "(STFM is all-context); use --gigapath-batch-size there.",
    )
    parser.add_argument(
        "--spatial-dir",
        default=None,
        help="(loki, stpath) Path to a Visium spatial/ folder. For loki: overrides spatial data in the input .h5ad (falls back to h5ad on read failure). For stpath: source folder for inline Gigapath feature extraction when --gigapath-h5 is not given, and a fallback for adata.obsm['spatial'] (loaded from tissue_positions.csv) when the input h5ad lacks spatial coordinates.",
    )
    parser.add_argument(
        "--housekeeping-genes",
        default=None,
        help="(loki) CSV with a 'genesymbol' column; genes to exclude from text encoding.",
    )
    parser.add_argument(
        "--library-id",
        default=None,
        help="(loki, stpath) Key under adata.uns['spatial'] (default: first).",
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
    parser.add_argument(
        "--gigapath-h5",
        default=None,
        help="(stpath) Path to a precomputed Gigapath sidecar .h5 with datasets "
             "'embeddings' [n, 1536] and 'barcodes' [n]. Optional: when omitted, "
             "the adapter computes Gigapath features inline and caches them at "
             "--gigapath-cache (default: <output>/gigapath_features.h5).",
    )
    parser.add_argument(
        "--fullres-image",
        default=None,
        help="(stpath) Full-resolution H&E image (TIFF/PNG/SVS) used for inline "
             "Gigapath feature extraction. Strongly preferred over the Visium "
             "hires PNG (Gigapath was trained on ~0.5 mpp tiles).",
    )
    parser.add_argument(
        "--patch-px",
        type=int,
        default=None,
        help="(stpath) Per-spot crop side length in image pixels for inline "
             "Gigapath features. Defaults to spot_diameter_fullres from Visium "
             "scalefactors (auto-scaled when falling back to the hires image).",
    )
    parser.add_argument(
        "--gigapath-batch-size",
        type=int,
        default=32,
        help="(stpath) Batch size for inline Gigapath inference (default: 32).",
    )
    parser.add_argument(
        "--gigapath-precision",
        default="fp32",
        choices=["fp32", "fp16"],
        help="(stpath) Precision for inline Gigapath inference (default: fp32).",
    )
    parser.add_argument(
        "--gigapath-cache",
        default=None,
        help="(stpath) Sidecar cache path written/read when --gigapath-h5 is "
             "not supplied (default: <output>/gigapath_features.h5).",
    )
    parser.add_argument(
        "--gigapath-recompute",
        action="store_true",
        help="(stpath) Force re-computation of the Gigapath sidecar .h5 even "
             "if a cache exists at --gigapath-cache. The sidecar is overwritten "
             "in place. Mutually exclusive with --gigapath-h5.",
    )
    parser.add_argument(
        "--organ-type",
        default="Others",
        help="(stpath) Organ token (default: Others). One of 25 values from "
             "src/models/stpath/utils/constants.py:organ_voc.",
    )
    parser.add_argument(
        "--tech-type",
        default=None,
        help="(stpath) Technology token. One of 'Spatial Transcriptomics', "
             "'Visium', 'Xenium', 'Visium HD'. Defaults to pad token.",
    )
    parser.add_argument(
        "--save-imputed-expression",
        action="store_true",
        help="(stpath) Also write imputed_expression.h5ad with the model's "
             "refined log1p expression on STPath's 38,984-gene vocabulary.",
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
            batch_size=args.batch_size if args.batch_size is not None else 64,
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
            batch_size=args.batch_size if args.batch_size is not None else 64,
            device=args.device,
        )
    elif args.model == "stpath":
        if args.batch_size is not None:
            print(
                "Warning: --batch-size is ignored for --model stpath "
                "(STFM is all-context). Use --gigapath-batch-size to "
                "tune inline Gigapath inference.",
                file=sys.stderr,
            )
        from src.adapters.stpath import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            gigapath_h5=args.gigapath_h5,
            spatial_dir=args.spatial_dir,
            fullres_image=args.fullres_image,
            library_id=args.library_id,
            patch_px=args.patch_px,
            gigapath_cache=args.gigapath_cache,
            gigapath_recompute=args.gigapath_recompute,
            gigapath_batch_size=args.gigapath_batch_size,
            gigapath_precision=args.gigapath_precision,
            organ_type=args.organ_type,
            tech_type=args.tech_type,
            save_imputed_expression=args.save_imputed_expression,
            device=args.device,
        )
    elif args.model == "stpath":
        if args.batch_size is not None:
            print(
                "Warning: --batch-size is ignored for --model stpath "
                "(STFM is all-context). Use --gigapath-batch-size to "
                "tune inline Gigapath inference.",
                file=sys.stderr,
            )
        from src.adapters.stpath import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            gigapath_h5=args.gigapath_h5,
            spatial_dir=args.spatial_dir,
            fullres_image=args.fullres_image,
            library_id=args.library_id,
            patch_px=args.patch_px,
            gigapath_cache=args.gigapath_cache,
            gigapath_recompute=args.gigapath_recompute,
            gigapath_batch_size=args.gigapath_batch_size,
            gigapath_precision=args.gigapath_precision,
            organ_type=args.organ_type,
            tech_type=args.tech_type,
            save_imputed_expression=args.save_imputed_expression,
            device=args.device,
        )
    elif args.model == "nicheformer":
        from src.adapters.nicheformer import run

        run(
            input_path=args.input,
            output_dir=args.output,
            model_dir=args.model_dir,
            technology=args.technology,
            batch_size=args.batch_size if args.batch_size is not None else 64,
            device=args.device,
            convert_symbols=args.convert_symbols,
            hgnc_mapping_path=args.hgnc_mapping,
        )


if __name__ == "__main__":
    main()
