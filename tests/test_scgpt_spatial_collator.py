"""Tests for src.models.scgpt_spatial.data_collator — DataCollator."""

import pytest
import torch

from src.models.scgpt_spatial.data_collator import DataCollator


def _make_example(genes, expressions):
    """Helper to create a single data example dict."""
    return {
        "id": torch.tensor(0),
        "genes": torch.tensor(genes, dtype=torch.long),
        "expressions": torch.tensor(expressions, dtype=torch.float),
    }


class TestDataCollator:
    def test_basic_collation(self):
        collator = DataCollator(
            do_padding=True,
            pad_token_id=0,
            pad_value=-2,
            do_mlm=False,
            do_binning=False,
            max_length=10,
            sampling=True,
            keep_first_n_tokens=1,
        )
        examples = [
            _make_example([1, 10, 20], [-2.0, 1.5, 2.5]),
            _make_example([1, 30], [-2.0, 3.0]),
        ]
        result = collator(examples)
        assert "gene" in result
        assert "expr" in result
        assert result["gene"].shape[0] == 2  # batch size
        assert result["gene"].shape[1] == 3  # max_ori_len=3, min(3,10)=3

    def test_padding_fills_correctly(self):
        collator = DataCollator(
            do_padding=True,
            pad_token_id=0,
            pad_value=-2,
            do_mlm=False,
            do_binning=False,
            max_length=5,
            sampling=True,
            keep_first_n_tokens=0,
        )
        examples = [
            _make_example([10, 20], [1.0, 2.0]),
        ]
        result = collator(examples)
        # Should be padded to length 2 (max_ori_len=2, min(2,5)=2)
        assert result["gene"].shape == (1, 2)

    def test_sampling_when_exceeds_max_length(self):
        collator = DataCollator(
            do_padding=True,
            pad_token_id=0,
            pad_value=-2,
            do_mlm=False,
            do_binning=False,
            max_length=3,
            sampling=True,
            keep_first_n_tokens=1,
        )
        # 5 tokens, max_length=3, keep_first=1 → keep [0] + sample 2 from [1..4]
        examples = [
            _make_example([1, 10, 20, 30, 40], [-2.0, 1.0, 2.0, 3.0, 4.0]),
        ]
        result = collator(examples)
        assert result["gene"].shape == (1, 3)
        # CLS token (first) should be preserved
        assert result["gene"][0, 0].item() == 1

    def test_binning(self):
        collator = DataCollator(
            do_padding=True,
            pad_token_id=0,
            pad_value=-2,
            do_mlm=False,
            do_binning=True,
            n_bins=51,
            max_length=10,
            sampling=True,
            keep_first_n_tokens=1,
        )
        examples = [
            _make_example([1, 10, 20, 30], [-2.0, 1.5, 2.5, 0.5]),
        ]
        result = collator(examples)
        expr = result["expr"]
        # After binning, non-pad values should be in [0, 50]
        # CLS token (position 0) has pad_value, kept unchanged
        assert expr[0, 0].item() == -2.0
        # Other positions should be binned (integers in [0, n_bins-1])
        for i in range(1, expr.shape[1]):
            val = expr[0, i].item()
            if val != -2.0:  # not padding
                assert 0 <= val <= 50

    def test_mlm_masking(self):
        torch.manual_seed(42)
        collator = DataCollator(
            do_padding=True,
            pad_token_id=0,
            pad_value=-2,
            do_mlm=True,
            do_binning=False,
            mlm_probability=0.5,
            mask_value=-1,
            max_length=10,
            sampling=True,
            keep_first_n_tokens=1,
        )
        examples = [
            _make_example([1, 10, 20, 30, 40, 50], [-2.0, 1.0, 2.0, 3.0, 4.0, 5.0]),
        ]
        result = collator(examples)
        assert "masked_expr" in result
        # Some values should be masked to -1
        masked = result["masked_expr"]
        expr = result["expr"]
        # CLS token should NOT be masked (keep_first_n_tokens=1)
        assert masked[0, 0].item() == expr[0, 0].item()

    def test_invalid_pad_config_raises(self):
        with pytest.raises(ValueError, match="pad_token_id"):
            DataCollator(do_padding=True, pad_token_id=None, max_length=10)

    def test_invalid_max_length_raises(self):
        with pytest.raises(ValueError, match="max_length"):
            DataCollator(do_padding=True, pad_token_id=0, max_length=None)
