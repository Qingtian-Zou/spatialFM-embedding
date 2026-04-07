"""Tests for src.models.scgpt_spatial.gene_tokenizer — GeneVocab dict wrapper."""

import json
import pickle

import numpy as np
import pytest
import torch

from src.models.scgpt_spatial.gene_tokenizer import (
    GeneVocab,
    tokenize_batch,
    pad_batch,
    tokenize_and_pad_batch,
    random_mask_value,
)
from tests.conftest import requires_model_weights, MODEL_DIR


# ======================================================================
# GeneVocab construction
# ======================================================================

class TestGeneVocabInit:
    def test_from_empty_list(self):
        v = GeneVocab([])
        assert len(v) == 0

    def test_from_gene_list(self, small_gene_list):
        v = GeneVocab(small_gene_list, default_token=None)
        assert len(v) == len(small_gene_list)
        for gene in small_gene_list:
            assert gene in v

    def test_from_gene_list_with_specials(self, small_gene_list):
        v = GeneVocab(small_gene_list, specials=["<pad>", "<cls>"], special_first=True)
        assert "<pad>" in v
        assert "<cls>" in v
        assert v["<pad>"] == 0
        assert v["<cls>"] == 1
        assert len(v) == len(small_gene_list) + 2

    def test_specials_last(self, small_gene_list):
        v = GeneVocab(small_gene_list, specials=["<pad>"], special_first=False, default_token=None)
        assert "<pad>" in v
        # When special_first=False, pad should be at the end
        assert v["<pad>"] == len(v) - 1

    def test_from_another_genevocab(self, vocab_with_specials):
        v2 = GeneVocab(vocab_with_specials)
        assert len(v2) == len(vocab_with_specials)
        for token in vocab_with_specials:
            assert v2[token] == vocab_with_specials[token]

    def test_copy_is_independent(self, vocab_with_specials):
        v2 = GeneVocab(vocab_with_specials)
        v2.append_token("NEW_GENE")
        assert "NEW_GENE" in v2
        assert "NEW_GENE" not in vocab_with_specials

    def test_specials_with_genevocab_raises(self, vocab_with_specials):
        with pytest.raises(ValueError, match="non-empty specials"):
            GeneVocab(vocab_with_specials, specials=["<mask>"])

    def test_invalid_input_type_raises(self):
        with pytest.raises(ValueError):
            GeneVocab({"TP53": 0})


# ======================================================================
# Core dict-like API
# ======================================================================

class TestGeneVocabDictAPI:
    def test_getitem_known_token(self, vocab_with_specials):
        idx = vocab_with_specials["TP53"]
        assert isinstance(idx, int)

    def test_getitem_unknown_no_default_raises(self):
        v = GeneVocab(["A", "B"], default_token=None)
        with pytest.raises(KeyError):
            v["UNKNOWN"]

    def test_getitem_unknown_with_default(self, vocab_with_specials):
        vocab_with_specials.set_default_index(999)
        assert vocab_with_specials["NONEXISTENT"] == 999

    def test_contains(self, vocab_with_specials):
        assert "TP53" in vocab_with_specials
        assert "NONEXISTENT" not in vocab_with_specials

    def test_call_batch_lookup(self, vocab_with_specials):
        result = vocab_with_specials(["TP53", "BRCA1", "<pad>"])
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(i, int) for i in result)

    def test_call_with_default_fallback(self, vocab_with_specials):
        vocab_with_specials.set_default_index(0)
        result = vocab_with_specials(["TP53", "NONEXISTENT"])
        assert result[1] == 0

    def test_len(self, vocab_with_specials):
        assert len(vocab_with_specials) == 8  # 5 genes + 3 specials

    def test_iter(self, vocab_with_specials):
        tokens = list(vocab_with_specials)
        assert len(tokens) == len(vocab_with_specials)
        assert "<pad>" in tokens


# ======================================================================
# Index management
# ======================================================================

class TestGeneVocabIndexMgmt:
    def test_set_get_default_index(self, vocab_with_specials):
        vocab_with_specials.set_default_index(42)
        assert vocab_with_specials.get_default_index() == 42

    def test_get_stoi(self, vocab_with_specials):
        stoi = vocab_with_specials.get_stoi()
        assert isinstance(stoi, dict)
        assert stoi["<pad>"] == vocab_with_specials["<pad>"]
        # Modifying the copy shouldn't affect original
        stoi["NEW"] = 999
        assert "NEW" not in vocab_with_specials

    def test_get_itos(self, vocab_with_specials):
        itos = vocab_with_specials.get_itos()
        assert isinstance(itos, list)
        assert len(itos) == len(vocab_with_specials)
        # Round-trip: itos[stoi[token]] == token
        for token in vocab_with_specials:
            assert itos[vocab_with_specials[token]] == token


# ======================================================================
# Token insertion
# ======================================================================

class TestGeneVocabInsertion:
    def test_insert_token(self):
        v = GeneVocab([], default_token=None)
        v.insert_token("A", 0)
        v.insert_token("B", 1)
        assert v["A"] == 0
        assert v["B"] == 1
        assert len(v) == 2

    def test_insert_duplicate_is_noop(self):
        v = GeneVocab([], default_token=None)
        v.insert_token("A", 0)
        v.insert_token("A", 5)  # should be ignored
        assert v["A"] == 0
        assert len(v) == 1

    def test_append_token(self, vocab_with_specials):
        old_len = len(vocab_with_specials)
        vocab_with_specials.append_token("NEW_GENE")
        assert len(vocab_with_specials) == old_len + 1
        assert vocab_with_specials["NEW_GENE"] == old_len

    def test_append_duplicate_is_noop(self, vocab_with_specials):
        old_len = len(vocab_with_specials)
        vocab_with_specials.append_token("TP53")
        assert len(vocab_with_specials) == old_len


# ======================================================================
# Pad token property
# ======================================================================

class TestGeneVocabPadToken:
    def test_pad_token_default_none(self):
        v = GeneVocab(["A"], default_token=None)
        assert v.pad_token is None

    def test_set_pad_token(self, vocab_with_specials):
        vocab_with_specials.pad_token = "<pad>"
        assert vocab_with_specials.pad_token == "<pad>"

    def test_set_invalid_pad_token_raises(self, vocab_with_specials):
        with pytest.raises(ValueError):
            vocab_with_specials.pad_token = "NONEXISTENT"

    def test_set_default_token(self, vocab_with_specials):
        vocab_with_specials.set_default_token("<cls>")
        # Should set the default index to <cls>'s index
        assert vocab_with_specials.get_default_index() == vocab_with_specials["<cls>"]

    def test_set_default_token_invalid_raises(self, vocab_with_specials):
        with pytest.raises(ValueError):
            vocab_with_specials.set_default_token("NONEXISTENT")


# ======================================================================
# Serialization
# ======================================================================

class TestGeneVocabSerialization:
    def test_from_json_file(self, vocab_json_file):
        v = GeneVocab.from_file(vocab_json_file)
        assert len(v) == 8
        assert v["<pad>"] == 0
        assert v["TP53"] == 3

    def test_from_dict(self):
        d = {"<pad>": 0, "<cls>": 1, "TP53": 2, "BRCA1": 3}
        v = GeneVocab.from_dict(d)
        assert len(v) == 4
        assert v["TP53"] == 2
        # Default token should be set to <pad>
        assert v.get_default_index() == 0

    def test_save_and_reload_json(self, vocab_with_specials, tmp_path):
        path = tmp_path / "test_vocab.json"
        vocab_with_specials.save_json(path)

        loaded = GeneVocab.from_file(path)
        assert len(loaded) == len(vocab_with_specials)
        for token in vocab_with_specials:
            assert loaded[token] == vocab_with_specials[token]

    def test_from_pkl_file(self, vocab_with_specials, tmp_path):
        path = tmp_path / "test_vocab.pkl"
        with open(path, "wb") as f:
            pickle.dump(vocab_with_specials, f)
        loaded = GeneVocab.from_file(path)
        assert len(loaded) == len(vocab_with_specials)

    def test_invalid_file_extension_raises(self, tmp_path):
        path = tmp_path / "bad.txt"
        path.write_text("test")
        with pytest.raises(ValueError, match="not a valid file type"):
            GeneVocab.from_file(path)

    @requires_model_weights
    def test_load_real_vocab(self):
        """Load the actual 60k-token vocab.json and verify basic properties."""
        v = GeneVocab.from_file(MODEL_DIR / "vocab.json")
        assert len(v) > 60000
        assert "<pad>" in v
        assert "<cls>" in v
        assert "TP53" in v
        # Round-trip
        stoi = v.get_stoi()
        assert stoi["TP53"] == v["TP53"]


# ======================================================================
# Build from iterator
# ======================================================================

class TestBuildVocabFromIterator:
    def test_basic_build(self):
        v = GeneVocab._build_vocab_from_iterator(["A", "B", "C"])
        assert len(v) == 3

    def test_build_with_specials_first(self):
        v = GeneVocab._build_vocab_from_iterator(
            ["A", "B"], specials=["<pad>"], special_first=True
        )
        assert v["<pad>"] == 0

    def test_duplicates_counted(self):
        v = GeneVocab._build_vocab_from_iterator(["A", "A", "B"])
        assert len(v) == 2  # A and B
        # A appears twice so should be first (higher frequency)
        assert v["A"] < v["B"]


# ======================================================================
# Tokenization utilities
# ======================================================================

class TestTokenizeBatch:
    def test_basic(self):
        data = np.array([[1.0, 0.0, 3.0], [0.0, 2.0, 0.0]], dtype=np.float32)
        gene_ids = np.array([10, 20, 30])
        result = tokenize_batch(data, gene_ids, append_cls=False, return_pt=True)
        assert len(result) == 2
        # First row: nonzero at idx 0, 2 → genes [10, 30]
        genes, values, _ = result[0]
        assert len(genes) == 2

    def test_with_cls(self):
        data = np.array([[1.0, 2.0]], dtype=np.float32)
        gene_ids = np.array([10, 20])
        result = tokenize_batch(data, gene_ids, append_cls=True, cls_id=99)
        genes, values, _ = result[0]
        assert genes[0].item() == 99  # CLS token first

    def test_shape_mismatch_raises(self):
        data = np.array([[1.0, 2.0]])
        gene_ids = np.array([10, 20, 30])
        with pytest.raises(ValueError, match="does not match"):
            tokenize_batch(data, gene_ids)


class TestPadBatch:
    def test_padding(self):
        from src.models.scgpt_spatial.gene_tokenizer import GeneVocab

        v = GeneVocab(["A"], specials=["<pad>"], default_token=None)
        batch = [
            (torch.tensor([1, 2, 3]), torch.tensor([0.1, 0.2, 0.3]), None),
            (torch.tensor([4, 5]), torch.tensor([0.4, 0.5]), None),
        ]
        result = pad_batch(batch, max_len=5, vocab=v, pad_token="<pad>", cls_appended=False)
        assert result["genes"].shape == (2, 3)  # max_ori_len=3, min(3,5)=3
        assert result["values"].shape == (2, 3)


class TestRandomMaskValue:
    def test_mask_preserves_shape(self):
        values = torch.tensor([[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 0.0, 0.0]])
        masked = random_mask_value(values, mask_ratio=0.5, mask_value=-1, pad_value=0)
        assert masked.shape == values.shape
        # Padding positions (0) should remain 0
        assert masked[0, 3].item() == 0.0
        assert masked[1, 2].item() == 0.0
        assert masked[1, 3].item() == 0.0
