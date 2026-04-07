from functools import lru_cache
import math
from typing import Optional

from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.transformer import _get_clones

try:
    from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func
    from flash_attn.bert_padding import unpad_input, pad_input
    from flash_attn.flash_attention import FlashAttention
    from flash_attn.modules.mha import FlashCrossAttention
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

from .layers import MultiheadAttention


def _require_flash_attn():
    if not _FLASH_ATTN_AVAILABLE:
        raise ImportError(
            "flash_attn is required for Flash attention layers but is not installed. "
            "Install it with: pip install flash-attn --no-build-isolation"
        )


class FlashscGPTMHA(nn.Module):
    """
    Custom MHA layer for scGPT. This takes two separate forward passes on the pect
    genes, and on the gen genes.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        bias=True,
        batch_first=True,
        attention_dropout=0.0,
        causal=False,
        device=None,
        dtype=None,
    ) -> None:
        _require_flash_attn()
        assert batch_first
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal

        self.num_heads = num_heads
        assert (
            self.embed_dim % num_heads == 0
        ), "self.kdim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        assert (
            self.head_dim % 8 == 0 and self.head_dim <= 128
        ), "Only support head_dim <= 128 and divisible by 8"

        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias, **factory_kwargs)
        self.self_attn = FlashAttention(attention_dropout=attention_dropout)
        self.cross_attn = MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=batch_first,
            **factory_kwargs,
        )
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
        need_weights=False,
    ):
        pcpt_qkv = self.Wqkv(pcpt_total_embs)

        pcpt_qkv = rearrange(
            pcpt_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        pcpt_context, pcpt_attn_weights = self.self_attn(
            pcpt_qkv,
            key_padding_mask=pcpt_key_padding_mask,
            need_weights=need_weights,
            causal=self.causal,
        )
        pcpt_context = self.out_proj(rearrange(pcpt_context, "b s h d -> b s (h d)"))

        if gen_total_embs is None:
            return (pcpt_context, None), (pcpt_attn_weights, None)

        gen_qkv = self.Wqkv(gen_total_embs)
        gen_qkv = rearrange(
            gen_qkv, "b s (three h d) -> b s three h d", three=3, h=self.num_heads
        )

        cross_q = gen_qkv[:, :, 0, :, :]
        cross_q = rearrange(cross_q, "b gen_s h d -> b gen_s (h d)")
        cross_kv = torch.cat(
            [pcpt_qkv[:, :, 1:, :, :], gen_qkv[:, :, 1:, :, :]], dim=1
        )
        cross_kv = rearrange(cross_kv, "b pcpt_gen_s two h d -> b pcpt_gen_s two (h d)")

        @lru_cache(maxsize=1)
        def make_mask(q_len, k_len, device):
            attention_mask = torch.zeros(
                (q_len, k_len), device=device, dtype=torch.bool
            )
            attention_mask[:, -q_len:] = ~torch.eye(
                q_len, device=device, dtype=torch.bool
            )
            return attention_mask

        attention_mask = make_mask(cross_q.shape[1], cross_kv.shape[1], cross_q.device)

        if pcpt_key_padding_mask is None and gen_key_padding_mask is None:
            key_padding_mask = None
        else:
            if pcpt_key_padding_mask is None:
                pcpt_key_padding_mask = torch.ones(
                    (pcpt_qkv.shape[0], pcpt_qkv.shape[1]),
                    device=pcpt_qkv.device,
                    dtype=torch.bool,
                )
            elif gen_key_padding_mask is None:
                gen_key_padding_mask = torch.ones(
                    (gen_qkv.shape[0], gen_qkv.shape[1]),
                    device=gen_qkv.device,
                    dtype=torch.bool,
                )
            key_padding_mask = ~torch.cat(
                [pcpt_key_padding_mask, gen_key_padding_mask], dim=1
            )
        cross_context, _ = self.cross_attn(
            cross_q,
            cross_kv[:, :, 0, :],
            cross_kv[:, :, 1, :],
            key_padding_mask=key_padding_mask,
            attn_mask=attention_mask,
        )
        gen_context = cross_context
        gen_attn_weights = None

        return (pcpt_context, gen_context), (pcpt_attn_weights, gen_attn_weights)


class FlashscGPTLayer(nn.Module):
    __constants__ = ["batch_first"]

    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        layer_norm_eps=1e-5,
        batch_first=True,
        device=None,
        dtype=None,
        norm_scheme="post",
    ) -> None:
        _require_flash_attn()
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.self_attn = FlashscGPTMHA(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=batch_first,
            attention_dropout=dropout,
            **factory_kwargs,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = self._get_activation_fn(activation)
        self.norm_scheme = norm_scheme
        if norm_scheme not in ["pre", "post"]:
            raise ValueError("norm_scheme must be either pre or post")

    @staticmethod
    def _get_activation_fn(activation):
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

    def __setstate__(self, state):
        if "activation" not in state:
            state["activation"] = F.relu
        super().__setstate__(state)

    def _reverse_key_padding_mask(self, src_key_padding_mask):
        if src_key_padding_mask is None:
            return None
        if not src_key_padding_mask.any().item():
            return None
        return ~src_key_padding_mask

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        pcpt_key_padding_mask_ = self._reverse_key_padding_mask(pcpt_key_padding_mask)
        gen_key_padding_mask_ = self._reverse_key_padding_mask(gen_key_padding_mask)

        if self.norm_scheme == "pre":
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            if gen_total_embs is not None:
                gen_total_embs = self.norm1(gen_total_embs)
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
        else:
            pcpt_total_embs2, gen_total_embs2 = self.self_attn(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask=pcpt_key_padding_mask_,
                gen_key_padding_mask=gen_key_padding_mask_,
            )[0]
            pcpt_total_embs = pcpt_total_embs + self.dropout1(pcpt_total_embs2)
            pcpt_total_embs = self.norm1(pcpt_total_embs)
            pcpt_total_embs2 = self.linear2(
                self.dropout(self.activation(self.linear1(pcpt_total_embs)))
            )
            pcpt_total_embs = pcpt_total_embs + self.dropout2(pcpt_total_embs2)
            pcpt_total_embs = self.norm2(pcpt_total_embs)

            if gen_total_embs is not None:
                gen_total_embs = gen_total_embs + self.dropout1(gen_total_embs2)
                gen_total_embs = self.norm1(gen_total_embs)
                gen_total_embs2 = self.linear2(
                    self.dropout(self.activation(self.linear1(gen_total_embs)))
                )
                gen_total_embs = gen_total_embs + self.dropout2(gen_total_embs2)
                gen_total_embs = self.norm2(gen_total_embs)

        return pcpt_total_embs, gen_total_embs


class FlashscGPTGenerator(nn.Module):
    __constants__ = ["norm"]

    def __init__(
        self,
        encoder_layer,
        num_layers,
        norm=None,
        mask_check=True,
    ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.mask_check = mask_check

    def forward(
        self,
        pcpt_total_embs: Tensor,
        gen_total_embs: Tensor,
        pcpt_key_padding_mask: Optional[Tensor] = None,
        gen_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        if pcpt_key_padding_mask is not None:
            _skpm_dtype = pcpt_key_padding_mask.dtype
            if _skpm_dtype != torch.bool and not torch.is_floating_point(
                pcpt_key_padding_mask
            ):
                raise AssertionError(
                    "only bool and floating types of key_padding_mask are supported"
                )

        for mod in self.layers:
            pcpt_total_embs, gen_total_embs = mod(
                pcpt_total_embs,
                gen_total_embs,
                pcpt_key_padding_mask,
                gen_key_padding_mask,
            )

        if self.norm is not None:
            pcpt_total_embs = self.norm(pcpt_total_embs)
            gen_total_embs = self.norm(gen_total_embs)

        return pcpt_total_embs, gen_total_embs
