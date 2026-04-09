"""Pure-PyTorch Nicheformer model for inference.

Ported from the PyTorch-Lightning reference at:
  references/nicheformer/src/nicheformer/models/_nicheformer.py

Layer names are kept identical to the reference so that the converted
state dict loads without key remapping.
"""

import json
import math
from pathlib import Path

import torch
import torch.nn as nn

from src.models.nicheformer.preprocess import complete_masking


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (non-learnable fallback)."""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        encoding = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        encoding = encoding.unsqueeze(0)
        self.register_buffer("encoding", encoding, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class NicheformerInference(nn.Module):
    """Nicheformer transformer encoder for embedding extraction.

    This is a pure ``nn.Module`` mirror of the original
    ``Nicheformer(pl.LightningModule)`` with training-only logic removed.
    All layer names match the Lightning version so that
    ``model.load_state_dict(ckpt['state_dict'])`` works directly.
    """

    def __init__(
        self,
        dim_model: int = 512,
        nheads: int = 16,
        dim_feedforward: int = 1024,
        nlayers: int = 12,
        dropout: float = 0.0,
        batch_first: bool = True,
        n_tokens: int = 20340,
        context_length: int = 1500,
        learnable_pe: bool = True,
        specie: bool = True,
        assay: bool = True,
        modality: bool = True,
        **kwargs,  # absorb extra hparams (lr, warmup, etc.)
    ):
        super().__init__()

        # Store config as plain attributes
        self.dim_model = dim_model
        self.nheads = nheads
        self.dim_feedforward = dim_feedforward
        self.nlayers = nlayers
        self.n_tokens = n_tokens
        self.context_length = context_length
        self.learnable_pe = learnable_pe
        self.specie = specie
        self.assay = assay
        self.modality = modality

        # --- Transformer encoder ---
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            batch_first=batch_first,
            dropout=dropout,
            layer_norm_eps=1e-12,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=nlayers,
            enable_nested_tensor=False,
        )

        # --- Embeddings ---
        self.embeddings = nn.Embedding(
            num_embeddings=n_tokens + 5,
            embedding_dim=dim_model,
            padding_idx=1,
        )

        if learnable_pe:
            self.positional_embedding = nn.Embedding(
                num_embeddings=context_length,
                embedding_dim=dim_model,
            )
            self.dropout = nn.Dropout(p=dropout)
            # pos is NOT in the checkpoint (it was a plain tensor in the
            # reference); register as buffer so it moves with .to(device).
            self.register_buffer(
                "pos", torch.arange(0, context_length, dtype=torch.long)
            )
        else:
            self.positional_embedding = PositionalEncoding(dim_model, context_length)

        # --- Heads (kept for state-dict compatibility, unused at inference) ---
        self.classifier_head = nn.Linear(dim_model, n_tokens, bias=False)
        self.classifier_head.bias = nn.Parameter(torch.zeros(n_tokens))
        self.pooler_head = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()
        self.cls_head = nn.Linear(dim_model, 164)

    # ------------------------------------------------------------------
    # Batch preparation (replaces Lightning on_after_batch_transfer hook)
    # ------------------------------------------------------------------

    def prepare_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Prepend auxiliary metadata tokens and truncate to context_length."""
        x = batch["X"]

        if self.modality and "modality" in batch:
            x = torch.cat((batch["modality"].reshape(-1, 1).to(torch.int32), x), dim=1)
        if self.assay and "assay" in batch:
            x = torch.cat((batch["assay"].reshape(-1, 1).to(torch.int32), x), dim=1)
        if self.specie and "specie" in batch:
            x = torch.cat((batch["specie"].reshape(-1, 1).to(torch.int32), x), dim=1)

        batch["X"] = x[:, : self.context_length]
        return batch

    # ------------------------------------------------------------------
    # Embedding extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_embeddings(
        self,
        batch: dict[str, torch.Tensor],
        layer: int = -1,
        with_context: bool = False,
    ) -> torch.Tensor:
        """Extract mean-pooled embeddings from a given transformer layer.

        Parameters
        ----------
        batch : dict
            Must contain ``'X'`` (int token indices).
        layer : int
            Transformer layer to extract from (default ``-1`` = last layer).
        with_context : bool
            If False (default), the first 3 context positions are stripped
            before mean-pooling.

        Returns
        -------
        torch.Tensor
            Shape ``(batch_size, dim_model)``.
        """
        batch = self.prepare_batch(batch)
        batch = complete_masking(batch, 0.0, self.n_tokens + 5)

        masked_indices = batch["masked_indices"]
        attention_mask = batch["attention_mask"]

        # Token + positional embeddings
        token_emb = self.embeddings(masked_indices)

        if self.learnable_pe:
            pos_emb = self.positional_embedding(self.pos.to(token_emb.device))
            emb = self.dropout(token_emb + pos_emb)
        else:
            emb = self.positional_embedding(token_emb)

        # Run through transformer layers up to the requested layer
        if layer < 0:
            layer = len(self.encoder.layers) + layer
        for i in range(layer + 1):
            emb = self.encoder.layers[i](
                emb, src_key_padding_mask=attention_mask, is_causal=False
            )

        # Strip context tokens and mean-pool
        if not with_context:
            emb = emb[:, 3:, :]

        return emb.mean(dim=1)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load_from_model_dir(cls, model_dir: str, device: str = "cpu"):
        """Load a converted Nicheformer checkpoint.

        Expects *model_dir* to contain ``hparams.json`` and
        ``model_state_dict.pt`` (produced by
        ``scripts/convert_nicheformer_ckpt.py``).
        """
        model_dir = Path(model_dir)

        with open(model_dir / "hparams.json") as f:
            hparams = json.load(f)

        model = cls(**hparams)

        state_dict = torch.load(
            model_dir / "model_state_dict.pt", map_location=device, weights_only=True
        )
        # strict=False: the checkpoint lacks 'pos' (registered buffer we added)
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        return model
