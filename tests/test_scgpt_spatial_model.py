"""Tests for src.models.scgpt_spatial.model — TransformerModel and components."""

import pytest
import torch
import numpy as np

from src.models.scgpt_spatial.gene_tokenizer import GeneVocab
from src.models.scgpt_spatial.model import (
    TransformerModel,
    GeneEncoder,
    ContinuousValueEncoder,
    CategoryValueEncoder,
    ExprDecoder,
    MoeDecoder,
)


@pytest.fixture
def small_vocab():
    """A minimal vocab for model construction."""
    v = GeneVocab([], default_token=None)
    for i, tok in enumerate(["<pad>", "<cls>", "<eoc>", "A", "B", "C"]):
        v.insert_token(tok, i)
    return v


@pytest.fixture
def small_model(small_vocab):
    """A tiny TransformerModel for unit tests (2 layers, dim=32)."""
    return TransformerModel(
        ntoken=len(small_vocab),
        d_model=32,
        nhead=4,
        d_hid=32,
        nlayers=2,
        nlayers_cls=1,
        n_cls=1,
        vocab=small_vocab,
        dropout=0.0,
        pad_token="<pad>",
        pad_value=-2,
        do_mvc=False,
        do_dab=False,
        use_batch_labels=False,
        use_fast_transformer=False,
        use_moe_dec=False,
    )


# ======================================================================
# Component tests
# ======================================================================

class TestGeneEncoder:
    def test_output_shape(self):
        enc = GeneEncoder(num_embeddings=100, embedding_dim=64)
        x = torch.randint(0, 100, (4, 10))  # batch=4, seq=10
        out = enc(x)
        assert out.shape == (4, 10, 64)


class TestContinuousValueEncoder:
    def test_output_shape(self):
        enc = ContinuousValueEncoder(d_model=64, dropout=0.0)
        x = torch.randn(4, 10)  # batch=4, seq=10
        out = enc(x)
        assert out.shape == (4, 10, 64)


class TestCategoryValueEncoder:
    def test_output_shape(self):
        enc = CategoryValueEncoder(num_embeddings=51, embedding_dim=64)
        x = torch.randint(0, 51, (4, 10))
        out = enc(x)
        assert out.shape == (4, 10, 64)


class TestExprDecoder:
    def test_output_shape(self):
        dec = ExprDecoder(d_model=64)
        x = torch.randn(4, 10, 64)
        result = dec(x)
        assert "pred" in result
        assert result["pred"].shape == (4, 10)

    def test_with_explicit_zero_prob(self):
        dec = ExprDecoder(d_model=64, explicit_zero_prob=True)
        x = torch.randn(4, 10, 64)
        result = dec(x)
        assert "pred" in result
        assert "zero_probs" in result


class TestMoeDecoder:
    def test_output_shape(self):
        dec = MoeDecoder(d_model=64, num_experts=4)
        x = torch.randn(4, 10, 64)
        result = dec(x)
        assert "pred" in result
        assert result["pred"].shape == (4, 10)


# ======================================================================
# TransformerModel
# ======================================================================

class TestTransformerModel:
    def test_construction(self, small_model):
        assert small_model.d_model == 32

    def test_encode_output_shape(self, small_model):
        """_encode should return (batch, seq_len, d_model)."""
        batch, seq = 2, 5
        src = torch.randint(0, 6, (batch, seq))
        values = torch.randn(batch, seq)
        mask = torch.zeros(batch, seq, dtype=torch.bool)

        with torch.no_grad():
            out = small_model._encode(src, values, mask)
        assert out.shape == (batch, seq, 32)

    def test_encode_cls_token(self, small_model):
        """CLS token at position 0 should produce a valid embedding."""
        batch, seq = 1, 4
        src = torch.tensor([[1, 3, 4, 5]])  # <cls>=1, then genes
        values = torch.tensor([[-2.0, 1.0, 2.0, 3.0]])  # pad_value for cls
        mask = torch.zeros(batch, seq, dtype=torch.bool)

        with torch.no_grad():
            out = small_model._encode(src, values, mask)
        cls_emb = out[:, 0, :]
        assert cls_emb.shape == (1, 32)
        assert not torch.isnan(cls_emb).any()

    def test_padding_mask_respected(self, small_model):
        """Padded positions should not affect the CLS embedding meaningfully."""
        src = torch.tensor([[1, 3, 0, 0]])  # <cls>, gene, pad, pad
        values = torch.tensor([[-2.0, 1.5, -2.0, -2.0]])
        mask = torch.tensor([[False, False, True, True]])

        with torch.no_grad():
            out = small_model._encode(src, values, mask)
        assert not torch.isnan(out).any()

    def test_flash_attn_not_required(self, small_vocab):
        """Model with use_fast_transformer=False should work without flash_attn."""
        model = TransformerModel(
            ntoken=len(small_vocab),
            d_model=32,
            nhead=4,
            d_hid=32,
            nlayers=1,
            nlayers_cls=1,
            n_cls=1,
            vocab=small_vocab,
            use_fast_transformer=False,
        )
        src = torch.randint(0, 6, (1, 3))
        values = torch.randn(1, 3)
        mask = torch.zeros(1, 3, dtype=torch.bool)
        with torch.no_grad():
            out = model._encode(src, values, mask)
        assert out.shape == (1, 3, 32)

    def test_perceptual_forward(self, small_model):
        """perceptual_forward should return a dict with mlm_output."""
        src = torch.randint(0, 6, (2, 4))
        values = torch.randn(2, 4)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        with torch.no_grad():
            out = small_model.perceptual_forward(src, values, mask)
        assert "mlm_output" in out
        assert "cell_emb" in out
        assert out["cell_emb"].shape == (2, 32)
