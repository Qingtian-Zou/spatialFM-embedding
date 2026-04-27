"""Tests for the Loki adapter — preprocessing helpers and Visium attach logic.

End-to-end shape tests that require the OmiCLIP checkpoint are gated behind
``requires_loki_weights``.
"""

import json

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from PIL import Image

from tests.conftest import (
    LOKI_MODEL_DIR,
    SAMPLE_H5AD,
    SAMPLE_SPATIAL_DIR,
    requires_loki_weights,
    requires_sample_data,
)


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=4, n_genes=80):
    rng = np.random.default_rng(0)
    X = sp.csr_matrix(rng.random((n_cells, n_genes), dtype=np.float32) * 10)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))


class TestGenerateGeneDF:
    def test_label_has_at_most_50_tokens(self):
        from src.models.loki.preprocess import generate_gene_df

        ad = _make_adata(n_cells=3, n_genes=80)
        hk = pd.DataFrame({"genesymbol": []})
        df = generate_gene_df(ad, hk, todense=True)

        assert list(df.columns) == ["label"]
        assert len(df) == 3
        for label in df["label"]:
            tokens = label.split()
            assert len(tokens) == 50

    def test_drops_dotted_and_dashed_genes(self):
        from src.models.loki.preprocess import generate_gene_df

        ad = _make_adata(n_cells=2, n_genes=60)
        ad.var.index = (
            ["A.1", "B-1"] + [f"GENE{i}" for i in range(58)]
        )
        hk = pd.DataFrame({"genesymbol": []})
        df = generate_gene_df(ad, hk, todense=True)
        for label in df["label"]:
            assert "A.1" not in label.split()
            assert "B-1" not in label.split()

    def test_excludes_housekeeping(self):
        from src.models.loki.preprocess import generate_gene_df

        ad = _make_adata(n_cells=2, n_genes=60)
        hk = pd.DataFrame({"genesymbol": ["GENE0", "GENE1"]})
        df = generate_gene_df(ad, hk, todense=True)
        for label in df["label"]:
            tokens = label.split()
            assert "GENE0" not in tokens
            assert "GENE1" not in tokens


class TestSegmentPatches:
    def test_writes_in_range_and_skips_out_of_range(self, tmp_path):
        from src.models.loki.preprocess import segment_patches

        img = np.zeros((100, 100, 3), dtype=np.uint8)
        coord = pd.DataFrame(
            {"pixel_x": [50, 5, 95], "pixel_y": [50, 95, 5]},
            index=["spot_in", "spot_oob_y", "spot_oob_x"],
        )
        segment_patches(img, coord, str(tmp_path), height=20, width=20)

        assert (tmp_path / "spot_in_hires.png").exists()
        assert not (tmp_path / "spot_oob_y_hires.png").exists()
        assert not (tmp_path / "spot_oob_x_hires.png").exists()


# ---------------------------------------------------------------------------
# Visium spatial attach (no model required)
# ---------------------------------------------------------------------------

class TestAttachVisium:
    def test_attach_from_synthetic_visium_dir(self, tmp_path):
        from src.adapters.loki import _attach_visium_spatial

        # Synthetic Visium folder
        spatial = tmp_path / "spatial"
        spatial.mkdir()
        Image.fromarray(np.zeros((50, 50, 3), dtype=np.uint8)).save(
            spatial / "tissue_hires_image.png"
        )
        (spatial / "scalefactors_json.json").write_text(
            json.dumps({"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 10})
        )
        pd.DataFrame(
            {
                "barcode": ["AAA-1", "BBB-1"],
                "in_tissue": [1, 1],
                "array_row": [0, 1],
                "array_col": [0, 1],
                "pxl_row_in_fullres": [10, 20],
                "pxl_col_in_fullres": [15, 25],
            }
        ).to_csv(spatial / "tissue_positions.csv", index=False)

        ad = anndata.AnnData(
            X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            obs=pd.DataFrame(index=["AAA-1", "BBB-1"]),
            var=pd.DataFrame(index=["G0", "G1", "G2"]),
        )
        _attach_visium_spatial(ad, str(spatial), library_id="loki")

        assert "spatial" in ad.obsm
        assert ad.obsm["spatial"].shape == (2, 2)
        # column 0 = pxl_col, column 1 = pxl_row
        np.testing.assert_array_equal(ad.obsm["spatial"][:, 0], [15, 25])
        np.testing.assert_array_equal(ad.obsm["spatial"][:, 1], [10, 20])
        assert "loki" in ad.uns["spatial"]
        assert ad.uns["spatial"]["loki"]["images"]["hires"].shape == (50, 50, 3)


# ---------------------------------------------------------------------------
# Resolve spatial — image dtype/channel normalization
# ---------------------------------------------------------------------------

def _adata_with_hires(hires):
    ad = anndata.AnnData(
        X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
        obs=pd.DataFrame(index=["AAA-1", "BBB-1"]),
        var=pd.DataFrame(index=["G0", "G1", "G2"]),
    )
    ad.obsm["spatial"] = np.array([[15, 10], [25, 20]])
    ad.uns["spatial"] = {
        "loki": {
            "images": {"hires": hires},
            "scalefactors": {"tissue_hires_scalef": 1.0, "spot_diameter_fullres": 10},
        }
    }
    return ad


class TestResolveSpatial:
    def test_passes_uint8_through(self):
        from src.adapters.loki import _resolve_spatial

        rng = np.random.default_rng(0)
        hires = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
        ad = _adata_with_hires(hires)

        img, _, _ = _resolve_spatial(ad, spatial_dir=None, library_id="loki")

        assert img.dtype == np.uint8
        np.testing.assert_array_equal(img, hires)

    def test_rescales_float01(self):
        from src.adapters.loki import _resolve_spatial

        hires = np.full((40, 50, 3), 0.5, dtype=np.float32)
        ad = _adata_with_hires(hires)

        img, _, _ = _resolve_spatial(ad, spatial_dir=None, library_id="loki")

        assert img.dtype == np.uint8
        # 0.5 * 255 = 127.5 → uint8 truncates to 127
        assert img.min() == 127 and img.max() == 127

    def test_handles_uint16(self):
        from src.adapters.loki import _resolve_spatial

        hires = np.full((40, 50, 3), 32768, dtype=np.uint16)
        ad = _adata_with_hires(hires)

        img, _, _ = _resolve_spatial(ad, spatial_dir=None, library_id="loki")

        assert img.dtype == np.uint8
        # 32768 / 256 = 128
        assert img.min() == 128 and img.max() == 128

    def test_strips_alpha(self):
        from src.adapters.loki import _resolve_spatial

        rng = np.random.default_rng(1)
        rgb = rng.integers(0, 256, size=(40, 50, 3), dtype=np.uint8)
        alpha = np.full((40, 50, 1), 255, dtype=np.uint8)
        hires = np.concatenate([rgb, alpha], axis=2)
        ad = _adata_with_hires(hires)

        img, _, _ = _resolve_spatial(ad, spatial_dir=None, library_id="loki")

        assert img.dtype == np.uint8
        assert img.shape == (40, 50, 3)
        np.testing.assert_array_equal(img, rgb)

    def test_clips_out_of_range_floats(self):
        from src.adapters.loki import _resolve_spatial

        hires = np.full((40, 50, 3), 0.5, dtype=np.float32)
        hires[0, 0] = -0.1
        hires[0, 1] = 1.5
        ad = _adata_with_hires(hires)

        img, _, _ = _resolve_spatial(ad, spatial_dir=None, library_id="loki")

        assert img.dtype == np.uint8
        assert img.min() == 0
        assert img.max() == 255
        # outliers saturate cleanly; bulk pixels still ~127
        assert (img[0, 0] == 0).all()
        assert (img[0, 1] == 255).all()
        assert img[1:, :].min() == 127 and img[1:, :].max() == 127


# ---------------------------------------------------------------------------
# End-to-end (requires real Loki weights)
# ---------------------------------------------------------------------------

@requires_loki_weights
@requires_sample_data
class TestLokiAdapterReal:
    def test_text_only_run(self, tmp_path):
        from src.adapters.loki import run

        out_dir = tmp_path / "out"
        adata = run(
            input_path=str(SAMPLE_H5AD),
            output_dir=str(out_dir),
            model_dir=str(LOKI_MODEL_DIR),
            device="cpu",
        )
        assert "X_loki_text" in adata.obsm
        assert adata.obsm["X_loki_text"].shape[1] == 768
        assert (out_dir / "embeddings_text.npy").exists()
        assert (out_dir / "embeddings.h5ad").exists()
        # text-only run should not produce image artifacts
        assert not (out_dir / "embeddings_image.npy").exists()

    def test_text_and_image_run(self, tmp_path):
        from src.adapters.loki import run

        if not SAMPLE_SPATIAL_DIR.exists():
            pytest.skip("sample Visium spatial dir missing")

        out_dir = tmp_path / "out"
        adata = run(
            input_path=str(SAMPLE_H5AD),
            output_dir=str(out_dir),
            model_dir=str(LOKI_MODEL_DIR),
            spatial_dir=str(SAMPLE_SPATIAL_DIR),
            patch_size=16,
            device="cpu",
        )
        assert "X_loki_text" in adata.obsm
        assert "X_loki_image" in adata.obsm
        assert adata.obsm["X_loki_image"].shape[1] == 768
        assert (out_dir / "embeddings_image.npy").exists()
