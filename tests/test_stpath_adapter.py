"""Tests for the STPath adapter — synthetic shape and validation tests.

End-to-end runs against the real 500 MB STPath checkpoint are gated behind
``requires_stpath_weights``.

Synthetic tests build a small STFM directly (matching the synthetic vocabulary
size) and write a fake ``stpath.pkl`` + ``symbol2ensembl.json`` so the adapter
can load weights and execute its preprocessing path without the real
checkpoint.
"""

import json
from pathlib import Path

import anndata
import h5py
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from src.adapters.stpath import _attach_spatial_from_dir, run as stpath_run
from src.models.stpath.gigapath import (
    resolve_he_inputs,
    save_gigapath_h5,
)
from src.models.stpath.model.model import STFM
from src.models.stpath.model.nn_utils.config import ModelConfig
from src.models.stpath.tokenization import (
    AnnotationTokenizer,
    GeneExpTokenizer,
    IDTokenizer,
    ImageTokenizer,
    TokenizerTools,
)

from tests.conftest import requires_stpath_weights, requires_sample_data


# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------

# Synthetic AnnData uses 200 gene symbols matched to the synthetic vocab so the
# adapter's coverage check (>= 100 mapped) passes. The synthetic STPath model is
# built with this same vocab size.
_N_SYNTH_GENES = 200
_FEATURE_DIM = 1536


def _synth_gene_symbols():
    return [f"SYNGENE{i:03d}" for i in range(_N_SYNTH_GENES)]


def _make_synthetic_model_dir(tmp_path):
    """Build a synthetic STPath model_dir with a tiny STFM checkpoint.

    Returns the model_dir path. The checkpoint matches the default config but
    with reduced n_genes/d_model for fast CPU execution.
    """
    model_dir = tmp_path / "stpath_synth"
    model_dir.mkdir()

    # Build symbol2ensembl.json with the synthetic vocab.
    symbol2ensembl = {
        f"SYNGENE{i:03d}": f"ENSG_SYN_{i:08d}" for i in range(_N_SYNTH_GENES)
    }
    (model_dir / "symbol2ensembl.json").write_text(json.dumps(symbol2ensembl))

    # Build the tokenizer to discover the actual n_tokens (n_genes + 2 specials).
    tokenizer = TokenizerTools(
        ge_tokenizer=GeneExpTokenizer(str(model_dir / "symbol2ensembl.json")),
        image_tokenizer=ImageTokenizer(feature_dim=_FEATURE_DIM),
        tech_tokenizer=IDTokenizer(id_type="tech"),
        specie_tokenizer=IDTokenizer(id_type="specie"),
        organ_tokenizer=IDTokenizer(id_type="organ"),
        cancer_anno_tokenizer=AnnotationTokenizer(id_type="disease"),
        domain_anno_tokenizer=AnnotationTokenizer(id_type="domain"),
    )

    # STPathInference.setup() hardcodes d_model=512 / n_layers=4 / n_heads=4
    # via ModelConfig.get_default_config(). The synthetic checkpoint must match
    # those dims so torch.load_state_dict succeeds.
    config = ModelConfig.get_default_config()
    config.feature_dim = _FEATURE_DIM
    config.activation = "gelu"
    config.n_genes = tokenizer.ge_tokenizer.n_tokens
    config.n_tech = tokenizer.tech_tokenizer.n_tokens
    config.n_species = tokenizer.specie_tokenizer.n_tokens
    config.n_organs = tokenizer.organ_tokenizer.n_tokens
    config.backbone = "spatial_transformer"

    torch.manual_seed(0)
    model = STFM(config)
    torch.save(model.state_dict(), model_dir / "stfm.pth")

    return model_dir


# Adapter output shape is the model's d_model (512 in upstream defaults).
_OUT_DIM = 512


def _make_synthetic_adata(n_spots=8, n_genes=_N_SYNTH_GENES, with_spatial=True, gene_overrides=None):
    rng = np.random.default_rng(42)
    counts = rng.integers(0, 50, size=(n_spots, n_genes)).astype(np.float32)
    X = sp.csr_matrix(counts)

    gene_names = gene_overrides if gene_overrides is not None else _synth_gene_symbols()[:n_genes]
    barcodes = [f"BC_{i:03d}" for i in range(n_spots)]
    adata = anndata.AnnData(
        X=X,
        obs=pd.DataFrame(index=barcodes),
        var=pd.DataFrame(index=gene_names),
    )
    if with_spatial:
        adata.obsm["spatial"] = rng.uniform(0, 1000, size=(n_spots, 2)).astype(np.float32)
    return adata


def _write_gigapath_h5(path, barcodes, n_spots=None, dim=_FEATURE_DIM):
    if n_spots is None:
        n_spots = len(barcodes)
    rng = np.random.default_rng(7)
    embeddings = rng.standard_normal((n_spots, dim)).astype(np.float32)
    coords = rng.uniform(0, 1000, size=(n_spots, 2)).astype(np.float32)
    barcodes_ascii = np.array(barcodes, dtype="S")
    with h5py.File(path, "w") as f:
        f.create_dataset("embeddings", data=embeddings)
        f.create_dataset("barcodes", data=barcodes_ascii)
        f.create_dataset("coords", data=coords)


# ---------------------------------------------------------------------------
# Validation / fail-fast tests
# ---------------------------------------------------------------------------

class TestValidation:
    def test_fails_when_spatial_missing(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(with_spatial=False)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        with pytest.raises(ValueError, match="obsm\\['spatial'\\]"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(gp),
                device="cpu",
            )

    def test_fails_when_gigapath_barcodes_mismatch(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata()
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        # Write gigapath with disjoint barcodes.
        _write_gigapath_h5(gp, ["DIFFERENT_BC_000", "DIFFERENT_BC_001"])

        with pytest.raises(ValueError, match="No barcodes shared"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(gp),
                device="cpu",
            )

    def test_fails_when_gene_coverage_too_low(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        # AnnData with gene names that are NOT in the synthetic vocab.
        adata = _make_synthetic_adata(
            gene_overrides=[f"FAKE_GENE_{i:03d}" for i in range(_N_SYNTH_GENES)]
        )
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        with pytest.raises(ValueError, match="Too few genes mapped"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(gp),
                device="cpu",
            )

    def test_fails_when_weights_missing(self, tmp_path):
        # Empty model_dir.
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        adata = _make_synthetic_adata()
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        with pytest.raises(FileNotFoundError, match="stfm\\.pth"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(empty_dir),
                gigapath_h5=str(gp),
                device="cpu",
            )

    def test_fails_when_gigapath_file_missing(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata()
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)

        with pytest.raises(FileNotFoundError, match="Gigapath sidecar not found"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(tmp_path / "nonexistent.h5"),
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Spatial-dir fallback — populates obsm["spatial"] from tissue_positions.csv
# ---------------------------------------------------------------------------

class TestSpatialDirFallback:
    def _write_tissue_positions_csv(self, spatial_dir, barcodes, rows, cols):
        spatial_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "barcode": barcodes,
                "in_tissue": 1,
                "array_row": np.arange(len(barcodes)),
                "array_col": np.arange(len(barcodes)),
                "pxl_row_in_fullres": rows,
                "pxl_col_in_fullres": cols,
            }
        ).to_csv(spatial_dir / "tissue_positions.csv", index=False)

    def test_helper_populates_spatial_with_col_row_order(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=5, with_spatial=False)
        rows = np.array([10, 20, 30, 40, 50])
        cols = np.array([100, 200, 300, 400, 500])
        spatial_dir = tmp_path / "spatial"
        self._write_tissue_positions_csv(spatial_dir, list(adata.obs_names), rows, cols)

        _attach_spatial_from_dir(adata, str(spatial_dir))

        assert "spatial" in adata.obsm
        assert adata.obsm["spatial"].shape == (5, 2)
        # Column 0 = pxl_col_in_fullres, column 1 = pxl_row_in_fullres.
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 0], cols)
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 1], rows)

    def test_helper_drops_unmatched_barcodes(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=5, with_spatial=False)
        # tissue_positions covers only barcodes 1..3 plus an extra unrelated one.
        present = list(adata.obs_names[1:4])
        extra = ["UNRELATED_BC"]
        rows = np.array([11, 22, 33, 99])
        cols = np.array([110, 220, 330, 990])
        spatial_dir = tmp_path / "spatial"
        self._write_tissue_positions_csv(spatial_dir, present + extra, rows, cols)

        _attach_spatial_from_dir(adata, str(spatial_dir))

        assert adata.n_obs == 3
        assert list(adata.obs_names) == present
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 0], [110, 220, 330])
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 1], [11, 22, 33])

    def test_helper_raises_on_no_shared_barcodes(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=3, with_spatial=False)
        spatial_dir = tmp_path / "spatial"
        self._write_tissue_positions_csv(
            spatial_dir,
            ["DISJOINT_0", "DISJOINT_1"],
            np.array([1, 2]),
            np.array([3, 4]),
        )

        with pytest.raises(ValueError, match="No shared barcodes"):
            _attach_spatial_from_dir(adata, str(spatial_dir))

    def test_helper_falls_back_to_legacy_headerless_csv(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=4, with_spatial=False)
        spatial_dir = tmp_path / "spatial"
        spatial_dir.mkdir()
        # Legacy v1: headerless tissue_positions_list.csv with fixed column order.
        rows = np.array([7, 8, 9, 10])
        cols = np.array([70, 80, 90, 100])
        df = pd.DataFrame(
            {
                "barcode": list(adata.obs_names),
                "in_tissue": 1,
                "array_row": np.arange(4),
                "array_col": np.arange(4),
                "pxl_row_in_fullres": rows,
                "pxl_col_in_fullres": cols,
            }
        )
        df.to_csv(spatial_dir / "tissue_positions_list.csv", header=False, index=False)

        _attach_spatial_from_dir(adata, str(spatial_dir))
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 0], cols)
        np.testing.assert_array_equal(adata.obsm["spatial"][:, 1], rows)

    def test_helper_raises_when_no_positions_file(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=3, with_spatial=False)
        spatial_dir = tmp_path / "empty_spatial"
        spatial_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="tissue_positions"):
            _attach_spatial_from_dir(adata, str(spatial_dir))

    def test_adapter_uses_fallback_when_spatial_missing(self, tmp_path):
        """End-to-end: input h5ad lacks obsm['spatial'] but --spatial-dir provides it."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4, with_spatial=False)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)

        spatial_dir = tmp_path / "spatial"
        rows = np.array([10, 20, 30, 40])
        cols = np.array([100, 200, 300, 400])
        self._write_tissue_positions_csv(
            spatial_dir, list(adata.obs_names), rows, cols
        )
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(tmp_path / "out"),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            spatial_dir=str(spatial_dir),
            device="cpu",
        )
        assert result.obsm["X_stpath"].shape == (4, _OUT_DIM)
        # Verify the loaded coords carry through (col, row).
        np.testing.assert_array_equal(result.obsm["spatial"][:, 0], cols)
        np.testing.assert_array_equal(result.obsm["spatial"][:, 1], rows)

    def test_existing_spatial_is_not_overridden_by_spatial_dir(self, tmp_path, monkeypatch):
        """If obsm['spatial'] is already populated, the fallback must not run."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4, with_spatial=True)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)

        spatial_dir = tmp_path / "spatial"
        self._write_tissue_positions_csv(
            spatial_dir,
            list(adata.obs_names),
            rows=np.array([1, 2, 3, 4]),
            cols=np.array([5, 6, 7, 8]),
        )
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        calls = {"attach": 0}

        def spy(*args, **kwargs):
            calls["attach"] += 1

        monkeypatch.setattr("src.adapters.stpath._attach_spatial_from_dir", spy)

        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(tmp_path / "out"),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            spatial_dir=str(spatial_dir),
            device="cpu",
        )
        assert calls["attach"] == 0
        assert result.obsm["X_stpath"].shape == (4, _OUT_DIM)

    def test_error_message_mentions_spatial_dir_when_both_missing(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4, with_spatial=False)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        with pytest.raises(ValueError, match="--spatial-dir"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(gp),
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Synthetic forward pass — requires no real weights
# ---------------------------------------------------------------------------

class TestSyntheticForwardPass:
    def test_produces_all_outputs(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=8)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        out_dir = tmp_path / "out"
        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            device="cpu",
        )

        # All four output formats exist.
        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()

        # X_stpath shape matches synthetic d_model=32.
        assert result.obsm["X_stpath"].shape == (8, _OUT_DIM)
        npy = np.load(out_dir / "embeddings.npy")
        assert npy.shape == (8, _OUT_DIM)

    def test_csv_and_tsv_parseable(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=5)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        out_dir = tmp_path / "out"
        stpath_run(
            input_path=str(h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            device="cpu",
        )

        df_csv = pd.read_csv(out_dir / "embeddings.csv", index_col=0)
        df_tsv = pd.read_csv(out_dir / "embeddings.tsv", sep="\t", index_col=0)
        assert df_csv.shape == (5, _OUT_DIM)
        assert df_tsv.shape == (5, _OUT_DIM)

    def test_save_imputed_expression_writes_sibling_file(self, tmp_path):
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        out_dir = tmp_path / "out"
        stpath_run(
            input_path=str(h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            save_imputed_expression=True,
            device="cpu",
        )

        imputed_path = out_dir / "imputed_expression.h5ad"
        assert imputed_path.exists()
        imputed_adata = anndata.read_h5ad(imputed_path)
        # Imputed expression columns = vocabulary size = _N_SYNTH_GENES (drops pad/mask).
        assert imputed_adata.shape == (4, _N_SYNTH_GENES)
        assert (out_dir / "imputed_expression.npy").exists()

    def test_organ_and_tech_token_lookup(self, tmp_path):
        """Run with valid organ + tech tokens — should not raise."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(tmp_path / "out"),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            organ_type="Kidney",
            tech_type="Visium",
            device="cpu",
        )
        assert result.obsm["X_stpath"].shape == (4, _OUT_DIM)

    def test_subsets_to_gigapath_intersection(self, tmp_path):
        """Adapter must align AnnData to the gigapath barcode set."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=10)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        # Gigapath has only 6 of the 10 barcodes.
        _write_gigapath_h5(gp, adata.obs_names.tolist()[:6])

        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(tmp_path / "out"),
            model_dir=str(model_dir),
            gigapath_h5=str(gp),
            device="cpu",
        )
        assert result.n_obs == 6
        assert result.obsm["X_stpath"].shape == (6, _OUT_DIM)


# ---------------------------------------------------------------------------
# Gigapath input resolution — no weights required
# ---------------------------------------------------------------------------

def _write_visium_spatial_dir(spatial_dir, n_spots=4, hires_size=128, scalef_hires=0.2):
    """Write a minimal Visium spatial/ folder under ``spatial_dir``."""
    spatial_dir.mkdir(parents=True)
    rng = np.random.default_rng(13)
    hires_img = rng.integers(0, 256, size=(hires_size, hires_size, 3), dtype=np.uint8)
    from PIL import Image as _PIL
    _PIL.fromarray(hires_img).save(spatial_dir / "tissue_hires_image.png")

    scalefactors = {
        "spot_diameter_fullres": 100.0,
        "tissue_hires_scalef": scalef_hires,
        "tissue_lowres_scalef": 0.05,
    }
    (spatial_dir / "scalefactors_json.json").write_text(json.dumps(scalefactors))

    # Fullres pixel coords; positions roughly within the (hires/scalef) frame.
    fullres_extent = int(hires_size / scalef_hires)
    barcodes = [f"BC_{i:03d}" for i in range(n_spots)]
    rows = rng.integers(50, fullres_extent - 50, size=n_spots)
    cols = rng.integers(50, fullres_extent - 50, size=n_spots)
    pd.DataFrame(
        {
            "barcode": barcodes,
            "in_tissue": 1,
            "array_row": np.arange(n_spots),
            "array_col": np.arange(n_spots),
            "pxl_row_in_fullres": rows,
            "pxl_col_in_fullres": cols,
        }
    ).to_csv(spatial_dir / "tissue_positions.csv", index=False)
    return barcodes, rows, cols, scalefactors


class TestGigapathInputResolution:
    def test_resolve_from_spatial_dir_hires(self, tmp_path):
        """Visium spatial dir without a fullres image should produce uint8 RGB
        + hires-scaled coords + a hires-scaled patch_px."""
        spatial_dir = tmp_path / "spatial"
        barcodes, rows, cols, scalef = _write_visium_spatial_dir(spatial_dir)

        adata = _make_synthetic_adata(n_spots=len(barcodes))
        adata.obs_names = pd.Index(barcodes)

        image, coord_df, patch_px, source = resolve_he_inputs(
            adata=adata,
            spatial_dir=str(spatial_dir),
            fullres_image=None,
            library_id=None,
            patch_px=None,
        )

        # uint8 RGB.
        assert image.dtype == np.uint8
        assert image.ndim == 3 and image.shape[2] == 3

        # Coords: hires-scaled (col, row).
        assert list(coord_df.columns) == ["pixel_x", "pixel_y"]
        np.testing.assert_allclose(
            coord_df["pixel_x"].to_numpy(),
            cols * scalef["tissue_hires_scalef"],
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            coord_df["pixel_y"].to_numpy(),
            rows * scalef["tissue_hires_scalef"],
            rtol=1e-6,
        )

        # patch_px = round(spot_diameter_fullres * tissue_hires_scalef)
        expected_patch = int(round(scalef["spot_diameter_fullres"] * scalef["tissue_hires_scalef"]))
        assert patch_px == expected_patch
        assert source.startswith("spatial-dir/hires")

    def test_resolve_with_explicit_fullres_image(self, tmp_path):
        """--fullres-image should be honoured; coords come from obsm['spatial']."""
        from PIL import Image as _PIL
        rng = np.random.default_rng(7)
        full_img = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
        full_path = tmp_path / "fullres.png"
        _PIL.fromarray(full_img).save(full_path)

        adata = _make_synthetic_adata(n_spots=3)
        # Patch_px must be supplied since adata has no scalefactors.
        image, coord_df, patch_px, source = resolve_he_inputs(
            adata=adata,
            spatial_dir=None,
            fullres_image=str(full_path),
            library_id=None,
            patch_px=42,
        )
        assert image.shape == (64, 96, 3)
        assert image.dtype == np.uint8
        assert patch_px == 42
        assert source == "fullres-image"
        # Coord df mirrors obsm['spatial'] one-to-one.
        np.testing.assert_allclose(
            coord_df["pixel_x"].to_numpy(),
            adata.obsm["spatial"][:, 0],
        )
        np.testing.assert_allclose(
            coord_df["pixel_y"].to_numpy(),
            adata.obsm["spatial"][:, 1],
        )

    def test_resolve_without_any_image_source_fails(self, tmp_path):
        adata = _make_synthetic_adata(n_spots=3)
        with pytest.raises(ValueError, match="Cannot locate an H&E image"):
            resolve_he_inputs(
                adata=adata,
                spatial_dir=None,
                fullres_image=None,
                library_id=None,
                patch_px=None,
            )

    def test_save_gigapath_h5_round_trips_through_adapter_reader(self, tmp_path):
        """Sidecar written by save_gigapath_h5 must be readable by the existing
        adapter reader and pass its barcode-decode path."""
        path = tmp_path / "gp.h5"
        embeddings = np.random.default_rng(0).standard_normal((5, _FEATURE_DIM)).astype(np.float32)
        barcodes = [f"BC_{i:03d}" for i in range(5)]
        coords = np.random.default_rng(1).uniform(0, 100, size=(5, 2)).astype(np.float32)
        save_gigapath_h5(str(path), embeddings, barcodes, coords)

        with h5py.File(path, "r") as f:
            assert f["embeddings"].shape == (5, _FEATURE_DIM)
            assert f["embeddings"].dtype == np.float32
            assert f["coords"].shape == (5, 2)
            raw = f["barcodes"][:]
        # Mirrors the decode at src/adapters/stpath.py:113-118.
        decoded = [b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b) for b in raw]
        assert decoded == barcodes


class TestInlineGigapathDispatch:
    """The adapter must reuse the cache when no --gigapath-h5 is given."""

    def test_reuses_cache_when_present(self, tmp_path):
        """Pre-seed <output>/gigapath_features.h5 and verify the adapter loads
        from it without invoking the encoder (no Gigapath weights needed)."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cache_path = out_dir / "gigapath_features.h5"
        _write_gigapath_h5(cache_path, adata.obs_names.tolist())

        # No --gigapath-h5 supplied: adapter must pick up the cache.
        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            device="cpu",
        )
        assert result.obsm["X_stpath"].shape == (4, _OUT_DIM)

    def test_recompute_flag_invalidates_cache(self, tmp_path, monkeypatch):
        """With gigapath_recompute=True the adapter must bypass the cache,
        invoke the inline encoder, and overwrite the sidecar in place."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        barcodes = adata.obs_names.tolist()

        out_dir = tmp_path / "out"
        out_dir.mkdir()
        cache_path = out_dir / "gigapath_features.h5"
        _write_gigapath_h5(cache_path, barcodes)
        with h5py.File(cache_path, "r") as f:
            seeded_embeddings = f["embeddings"][:].copy()

        fake_embeddings = np.full((len(barcodes), _FEATURE_DIM), 0.5, dtype=np.float32)
        fake_coord_df = pd.DataFrame(
            {
                "pixel_x": np.arange(len(barcodes), dtype=np.float32),
                "pixel_y": np.arange(len(barcodes), dtype=np.float32),
            },
            index=barcodes,
        )
        calls = {"resolve": 0, "compute": 0}

        def fake_resolve(**kwargs):
            calls["resolve"] += 1
            return np.zeros((1, 1, 3), dtype=np.uint8), fake_coord_df, 16, "fake"

        def fake_compute(**kwargs):
            calls["compute"] += 1
            return fake_embeddings, list(barcodes)

        monkeypatch.setattr("src.adapters.stpath.resolve_he_inputs", fake_resolve)
        monkeypatch.setattr("src.adapters.stpath.compute_gigapath_features", fake_compute)

        result = stpath_run(
            input_path=str(h5ad),
            output_dir=str(out_dir),
            model_dir=str(model_dir),
            gigapath_recompute=True,
            device="cpu",
        )

        assert calls["resolve"] == 1, "resolve_he_inputs not invoked despite force flag"
        assert calls["compute"] == 1, "compute_gigapath_features not invoked despite force flag"
        assert result.obsm["X_stpath"].shape == (4, _OUT_DIM)

        with h5py.File(cache_path, "r") as f:
            recomputed = f["embeddings"][:]
        assert np.allclose(recomputed, fake_embeddings)
        assert not np.allclose(recomputed, seeded_embeddings)

    def test_recompute_with_explicit_sidecar_raises(self, tmp_path):
        """--gigapath-h5 + --gigapath-recompute is contradictory; fail fast."""
        model_dir = _make_synthetic_model_dir(tmp_path)
        adata = _make_synthetic_adata(n_spots=4)
        h5ad = tmp_path / "in.h5ad"
        adata.write_h5ad(h5ad)
        gp = tmp_path / "gp.h5"
        _write_gigapath_h5(gp, adata.obs_names.tolist())

        with pytest.raises(ValueError, match="mutually exclusive"):
            stpath_run(
                input_path=str(h5ad),
                output_dir=str(tmp_path / "out"),
                model_dir=str(model_dir),
                gigapath_h5=str(gp),
                gigapath_recompute=True,
                device="cpu",
            )


# ---------------------------------------------------------------------------
# Real-data tests — gated by weights and sample data availability
# ---------------------------------------------------------------------------

@requires_stpath_weights
@requires_sample_data
class TestAdapterReal:
    def test_real_data_smoke(self):  # pragma: no cover (needs weights + gigapath sidecar)
        # Intentionally skipped in CI. Developers should run end-to-end with
        # real example data via:
        #   pyenv exec python src/embed.py --model stpath \
        #     --input references/STPath/example_data/INT2.h5ad \
        #     --gigapath-h5 references/STPath/example_data/INT2.h5 \
        #     --output output/stpath_int2 \
        #     --model-dir model_weights/stpath/ \
        #     --organ-type Kidney --tech-type Visium --device cuda
        pytest.skip("End-to-end real-data test requires a paired Gigapath sidecar; run manually.")
