"""Tests for the Nicheformer adapter — preprocessing, tokenization, model, and
end-to-end embedding extraction.

Unit tests use synthetic data and random weights.  End-to-end tests that
require the converted checkpoint are gated behind ``requires_nicheformer_weights``.
"""

from pathlib import Path

import anndata
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
import torch

from tests.conftest import (
    NICHEFORMER_MODEL_DIR,
    SAMPLE_H5AD,
    requires_nicheformer_weights,
    requires_sample_data,
)

BUNDLED_HGNC_TSV = (
    Path(__file__).resolve().parent.parent
    / "src" / "models" / "nicheformer" / "HGNC_symbol_all_genes.tsv"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_adata(n_cells=20, n_genes=100):
    """Synthetic AnnData with positive counts."""
    rng = np.random.default_rng(42)
    X = sp.csr_matrix(rng.random((n_cells, n_genes), dtype=np.float32) * 100)
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))


def _make_dense_adata(n_cells=10, n_genes=50):
    rng = np.random.default_rng(7)
    X = rng.random((n_cells, n_genes), dtype=np.float32) * 100
    gene_names = [f"GENE{i}" for i in range(n_genes)]
    return anndata.AnnData(X=X, var=pd.DataFrame(index=gene_names))


# ---------------------------------------------------------------------------
# sf_normalize
# ---------------------------------------------------------------------------

class TestSfNormalize:
    def test_output_sums_to_10000_sparse(self):
        from src.models.nicheformer.preprocess import sf_normalize

        ad = _make_adata(n_cells=5, n_genes=50)
        normed = sf_normalize(ad.X)
        row_sums = np.array(normed.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 10_000.0, rtol=1e-5)

    def test_output_sums_to_10000_dense(self):
        from src.models.nicheformer.preprocess import sf_normalize

        ad = _make_dense_adata()
        normed = sf_normalize(ad.X)
        row_sums = normed.sum(axis=1)
        np.testing.assert_allclose(row_sums, 10_000.0, rtol=1e-5)

    def test_zero_row_no_error(self):
        from src.models.nicheformer.preprocess import sf_normalize

        X = np.zeros((3, 5), dtype=np.float32)
        X[1, :] = [1, 2, 3, 4, 5]
        normed = sf_normalize(X)
        assert normed[0].sum() == 0.0
        np.testing.assert_allclose(normed[1].sum(), 10_000.0, rtol=1e-5)


# ---------------------------------------------------------------------------
# tokenize_data
# ---------------------------------------------------------------------------

class TestTokenizeData:
    def test_output_shape(self):
        from src.models.nicheformer.preprocess import tokenize_data

        rng = np.random.default_rng(0)
        X = rng.random((8, 50), dtype=np.float32) * 10
        med = rng.random(50, dtype=np.float64) + 0.1
        tokens = tokenize_data(X, med, max_seq_len=4096)
        assert tokens.shape == (8, 4096)
        assert tokens.dtype == np.int32

    def test_token_offset(self):
        from src.models.nicheformer.preprocess import tokenize_data

        rng = np.random.default_rng(1)
        X = rng.random((4, 20), dtype=np.float32) * 10
        med = np.ones(20, dtype=np.float64)
        tokens = tokenize_data(X, med, max_seq_len=100)
        # All non-padding tokens must be >= aux_tokens (30)
        nonzero = tokens[tokens != 0]
        assert (nonzero >= 30).all()

    def test_padding_is_zero(self):
        from src.models.nicheformer.preprocess import tokenize_data

        # 3 non-zero genes → at most 3 tokens, rest should be 0
        X = np.zeros((2, 100), dtype=np.float32)
        X[0, 0] = 5.0
        X[0, 1] = 3.0
        X[0, 2] = 1.0
        med = np.ones(100, dtype=np.float64)
        tokens = tokenize_data(X, med, max_seq_len=50)
        assert (tokens[0, 3:] == 0).all()
        assert (tokens[1, :] == 0).all()  # all-zero cell


# ---------------------------------------------------------------------------
# complete_masking
# ---------------------------------------------------------------------------

class TestCompleteMasking:
    def test_no_masking_at_p_zero(self):
        from src.models.nicheformer.preprocess import complete_masking

        X = torch.tensor([[30, 35, 40, 0, 0]], dtype=torch.int64)
        batch = {"X": X.clone()}
        out = complete_masking(batch, 0.0, 20345)

        # Padding 0s become 1s in X
        expected_x = torch.tensor([[30, 35, 40, 1, 1]], dtype=torch.int64)
        torch.testing.assert_close(out["X"], expected_x)
        # masked_indices should equal remapped X (no masking at p=0)
        torch.testing.assert_close(out["masked_indices"], expected_x)

    def test_attention_mask_marks_padding(self):
        from src.models.nicheformer.preprocess import complete_masking

        X = torch.tensor([[30, 35, 0, 0]], dtype=torch.int64)
        batch = {"X": X.clone()}
        out = complete_masking(batch, 0.0, 20345)
        expected_mask = torch.tensor([[False, False, True, True]])
        torch.testing.assert_close(out["attention_mask"], expected_mask)


# ---------------------------------------------------------------------------
# NicheformerDataset
# ---------------------------------------------------------------------------

class TestNicheformerDataset:
    def test_length_matches_adata(self):
        from src.models.nicheformer.preprocess import NicheformerDataset

        ad = _make_adata(n_cells=12, n_genes=30)
        tech_mean = np.ones(30, dtype=np.float64)
        ds = NicheformerDataset(ad, tech_mean, max_seq_len=100)
        assert len(ds) == 12

    def test_item_has_X_key(self):
        from src.models.nicheformer.preprocess import NicheformerDataset

        ad = _make_adata(n_cells=5, n_genes=20)
        tech_mean = np.ones(20, dtype=np.float64)
        ds = NicheformerDataset(ad, tech_mean, max_seq_len=50)
        item = ds[0]
        assert "X" in item
        assert item["X"].shape == (50,)
        assert item["X"].dtype in (torch.int32, torch.int64)


# ---------------------------------------------------------------------------
# Model construction and forward pass (random weights)
# ---------------------------------------------------------------------------

class TestNicheformerModel:
    def test_model_construction(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=64, nheads=4, dim_feedforward=128, nlayers=2,
            n_tokens=100, context_length=50,
        )
        assert model.dim_model == 64
        assert model.nlayers == 2

    def test_get_embeddings_shape(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=64, nheads=4, dim_feedforward=128, nlayers=2,
            n_tokens=100, context_length=50,
            specie=False, assay=False, modality=False,
        )
        model.eval()

        # Synthetic batch: 4 cells, seq_len=50, tokens in range [30, 100+5)
        rng = np.random.default_rng(42)
        tokens = rng.integers(30, 105, size=(4, 50), dtype=np.int32)
        tokens[:, 10:] = 0  # pad most positions
        batch = {"X": torch.tensor(tokens, dtype=torch.long)}

        emb = model.get_embeddings(batch)
        assert emb.shape == (4, 64)

    def test_get_embeddings_with_context_tokens(self):
        from src.models.nicheformer.model import NicheformerInference

        model = NicheformerInference(
            dim_model=32, nheads=4, dim_feedforward=64, nlayers=1,
            n_tokens=100, context_length=20,
            specie=True, assay=True, modality=True,
        )
        model.eval()

        batch = {
            "X": torch.zeros(2, 20, dtype=torch.long),
            "specie": torch.tensor([5, 5], dtype=torch.int64),
            "assay": torch.tensor([6, 6], dtype=torch.int64),
            "modality": torch.tensor([7, 7], dtype=torch.int64),
        }
        batch["X"][:, :5] = torch.tensor([30, 31, 32, 33, 34])

        emb = model.get_embeddings(batch)
        assert emb.shape == (2, 32)


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

class TestAlignGenes:
    def test_shared_genes_subset(self):
        from src.models.nicheformer.preprocess import align_genes

        ad = anndata.AnnData(
            X=sp.csr_matrix(np.ones((3, 5), dtype=np.float32)),
            var=pd.DataFrame(index=["A", "B", "C", "D", "E"]),
        )
        vocab = ["B", "D", "F", "A"]
        tech_mean = np.array([1.0, 2.0, 3.0, 4.0])

        aligned_ad, aligned_mean = align_genes(ad, vocab, tech_mean)
        assert list(aligned_ad.var_names) == ["B", "D", "A"]
        np.testing.assert_array_equal(aligned_mean, [1.0, 2.0, 4.0])

    def test_no_overlap_raises(self):
        from src.models.nicheformer.preprocess import align_genes

        ad = anndata.AnnData(
            X=sp.csr_matrix(np.ones((2, 3), dtype=np.float32)),
            var=pd.DataFrame(index=["X", "Y", "Z"]),
        )
        with pytest.raises(ValueError, match="No input genes match"):
            align_genes(ad, ["A", "B"], np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# Ensembl version stripping
# ---------------------------------------------------------------------------

class TestStripEnsemblVersion:
    def test_strips_version_suffix(self):
        from src.models.nicheformer.preprocess import _strip_ensembl_version
        assert _strip_ensembl_version("ENSG00000141510.18") == "ENSG00000141510"

    def test_no_version_unchanged(self):
        from src.models.nicheformer.preprocess import _strip_ensembl_version
        assert _strip_ensembl_version("ENSG00000141510") == "ENSG00000141510"

    def test_non_ensembl_unchanged(self):
        from src.models.nicheformer.preprocess import _strip_ensembl_version
        assert _strip_ensembl_version("TP53") == "TP53"
        assert _strip_ensembl_version("TP53.5") == "TP53.5"


# ---------------------------------------------------------------------------
# Ensembl detection heuristic
# ---------------------------------------------------------------------------

class TestLooksLikeEnsembl:
    def test_all_ensembl_returns_true(self):
        from src.models.nicheformer.preprocess import _looks_like_ensembl
        assert _looks_like_ensembl(["ENSG00000141510", "ENSG00000012048"])

    def test_all_symbols_returns_false(self):
        from src.models.nicheformer.preprocess import _looks_like_ensembl
        assert not _looks_like_ensembl(["TP53", "BRCA1", "EGFR"])

    def test_50_50_at_default_threshold(self):
        from src.models.nicheformer.preprocess import _looks_like_ensembl
        assert _looks_like_ensembl(["ENSG00000141510", "TP53"])

    def test_below_threshold_returns_false(self):
        from src.models.nicheformer.preprocess import _looks_like_ensembl
        assert not _looks_like_ensembl(
            ["ENSG00000141510", "TP53", "BRCA1", "EGFR"]
        )

    def test_empty_returns_false(self):
        from src.models.nicheformer.preprocess import _looks_like_ensembl
        assert not _looks_like_ensembl([])


# ---------------------------------------------------------------------------
# HGNC TSV parsing
# ---------------------------------------------------------------------------

def _write_mini_hgnc_tsv(path: Path) -> None:
    """Write a tiny HGNC TSV with the columns the loader needs."""
    rows = [
        # Approved row
        ["HGNC:1", "GENEA", "Gene A", "Approved", "OLDA1, OLDA2", "ALA1",
         "1", "", "", "", "1", "ENSG00000000001"],
        # Approved row whose Approved symbol collides with another row's alias
        ["HGNC:2", "GENEB", "Gene B", "Approved", "", "GENEA",
         "2", "", "", "", "2", "ENSG00000000002"],
        # Withdrawn — must be skipped
        ["HGNC:3", "GENEW", "withdrawn", "Symbol Withdrawn", "", "",
         "3", "", "", "", "3", "ENSG00000000003"],
        # Approved but missing Ensembl — must be skipped
        ["HGNC:4", "GENEN", "Gene N", "Approved", "", "",
         "4", "", "", "", "4", ""],
    ]
    cols = ["HGNC ID", "Approved symbol", "Approved name", "Status",
            "Previous symbols", "Alias symbols", "Chromosome",
            "Accession numbers", "RefSeq IDs", "Enzyme IDs",
            "NCBI Gene ID", "Ensembl gene ID"]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, sep="\t", index=False)


class TestLoadHgncMapping:
    def test_parses_approved_symbols(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        tsv = tmp_path / "mini_hgnc.tsv"
        _write_mini_hgnc_tsv(tsv)
        mapping = _load_hgnc_mapping(str(tsv))
        assert mapping["GENEA"] == "ENSG00000000001"
        assert mapping["GENEB"] == "ENSG00000000002"

    def test_aliases_and_previous_resolve(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        tsv = tmp_path / "mini_hgnc.tsv"
        _write_mini_hgnc_tsv(tsv)
        mapping = _load_hgnc_mapping(str(tsv))
        assert mapping["OLDA1"] == "ENSG00000000001"
        assert mapping["OLDA2"] == "ENSG00000000001"
        assert mapping["ALA1"] == "ENSG00000000001"

    def test_approved_beats_alias_on_collision(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        tsv = tmp_path / "mini_hgnc.tsv"
        _write_mini_hgnc_tsv(tsv)
        mapping = _load_hgnc_mapping(str(tsv))
        # GENEA is GENEB's alias, but also an approved symbol — approved wins.
        assert mapping["GENEA"] == "ENSG00000000001"

    def test_withdrawn_row_skipped(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        tsv = tmp_path / "mini_hgnc.tsv"
        _write_mini_hgnc_tsv(tsv)
        mapping = _load_hgnc_mapping(str(tsv))
        assert "GENEW" not in mapping

    def test_empty_ensembl_skipped(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        tsv = tmp_path / "mini_hgnc.tsv"
        _write_mini_hgnc_tsv(tsv)
        mapping = _load_hgnc_mapping(str(tsv))
        assert "GENEN" not in mapping

    def test_missing_columns_raises(self, tmp_path):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        bad = tmp_path / "bad.tsv"
        pd.DataFrame({"foo": ["bar"]}).to_csv(bad, sep="\t", index=False)
        with pytest.raises(ValueError, match="missing columns"):
            _load_hgnc_mapping(str(bad))

    @pytest.mark.skipif(
        not BUNDLED_HGNC_TSV.exists(),
        reason="bundled HGNC TSV not present",
    )
    def test_real_tsv_known_mappings(self):
        from src.models.nicheformer.preprocess import _load_hgnc_mapping
        mapping = _load_hgnc_mapping(str(BUNDLED_HGNC_TSV))
        assert mapping["TP53"] == "ENSG00000141510"
        assert mapping["BRCA1"] == "ENSG00000012048"
        assert mapping["A1BG"] == "ENSG00000121410"


# ---------------------------------------------------------------------------
# HGNC symbol -> Ensembl conversion
# ---------------------------------------------------------------------------

def _toy_mapping() -> dict:
    return {
        "TP53": "ENSG00000141510",
        "BRCA1": "ENSG00000012048",
        "EGFR": "ENSG00000146648",
        # An alias that collapses onto TP53's Ensembl ID.
        "P53": "ENSG00000141510",
    }


class TestToEnsemblIds:
    def test_symbols_converted(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        ad = anndata.AnnData(
            X=np.ones((2, 3), dtype=np.float32),
            var=pd.DataFrame(index=["TP53", "BRCA1", "EGFR"]),
        )
        out = to_ensembl_ids(ad, _toy_mapping())
        assert list(out.var_names) == [
            "ENSG00000141510", "ENSG00000012048", "ENSG00000146648"
        ]
        assert out.shape == (2, 3)

    def test_ensembl_passthrough(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        ad = anndata.AnnData(
            X=np.ones((2, 2), dtype=np.float32),
            var=pd.DataFrame(index=["ENSG00000141510", "ENSG00000012048"]),
        )
        out = to_ensembl_ids(ad, {})
        assert list(out.var_names) == ["ENSG00000141510", "ENSG00000012048"]

    def test_unmapped_dropped(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        ad = anndata.AnnData(
            X=np.ones((2, 4), dtype=np.float32),
            var=pd.DataFrame(index=["TP53", "FAKEGENE", "BRCA1", "BOGUS"]),
        )
        out = to_ensembl_ids(ad, _toy_mapping())
        assert list(out.var_names) == ["ENSG00000141510", "ENSG00000012048"]
        assert out.shape == (2, 2)

    def test_case_insensitive(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        ad = anndata.AnnData(
            X=np.ones((1, 2), dtype=np.float32),
            var=pd.DataFrame(index=["tp53", "Brca1"]),
        )
        out = to_ensembl_ids(ad, _toy_mapping())
        assert list(out.var_names) == ["ENSG00000141510", "ENSG00000012048"]

    def test_duplicates_summed_dense(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        # TP53 and P53 both map to ENSG00000141510 -> columns should be summed.
        X = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]], dtype=np.float32)
        ad = anndata.AnnData(
            X=X.copy(),
            var=pd.DataFrame(index=["TP53", "P53", "BRCA1"]),
        )
        out = to_ensembl_ids(ad, _toy_mapping())
        assert list(out.var_names) == ["ENSG00000141510", "ENSG00000012048"]
        assert out.shape == (2, 2)
        out_X = np.asarray(out.X)
        np.testing.assert_array_equal(out_X[:, 0], X[:, 0] + X[:, 1])
        np.testing.assert_array_equal(out_X[:, 1], X[:, 2])

    def test_duplicates_summed_sparse(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        X = sp.csr_matrix(
            np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]], dtype=np.float32)
        )
        ad = anndata.AnnData(
            X=X,
            var=pd.DataFrame(index=["TP53", "P53", "BRCA1"]),
        )
        out = to_ensembl_ids(ad, _toy_mapping())
        assert list(out.var_names) == ["ENSG00000141510", "ENSG00000012048"]
        out_dense = out.X.toarray() if sp.issparse(out.X) else np.asarray(out.X)
        np.testing.assert_array_equal(out_dense[:, 0], [3.0, 7.0])
        np.testing.assert_array_equal(out_dense[:, 1], [10.0, 20.0])

    def test_zero_mapped_raises_with_mouse_hint(self):
        from src.models.nicheformer.preprocess import to_ensembl_ids
        ad = anndata.AnnData(
            X=np.ones((1, 2), dtype=np.float32),
            var=pd.DataFrame(index=["Trp53", "Brca1"]),
        )
        with pytest.raises(ValueError, match="mouse"):
            to_ensembl_ids(ad, {})


# ---------------------------------------------------------------------------
# End-to-end (requires converted Nicheformer weights)
# ---------------------------------------------------------------------------

@requires_nicheformer_weights
@requires_sample_data
class TestNicheformerAdapterReal:
    def test_run_produces_embeddings(self, tmp_path):
        from src.adapters.nicheformer import run

        out_dir = tmp_path / "out"
        adata = run(
            input_path=str(SAMPLE_H5AD),
            output_dir=str(out_dir),
            model_dir=str(NICHEFORMER_MODEL_DIR),
            technology="merfish",
            batch_size=32,
            device="cpu",
        )
        assert "X_nicheformer" in adata.obsm
        assert adata.obsm["X_nicheformer"].shape[1] == 512
        assert (out_dir / "embeddings.npy").exists()
        assert (out_dir / "embeddings.h5ad").exists()
        assert (out_dir / "embeddings.csv").exists()
        assert (out_dir / "embeddings.tsv").exists()
