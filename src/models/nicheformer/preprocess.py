"""Nicheformer preprocessing, tokenization, and dataset utilities.

Ported from the reference implementation at:
  references/nicheformer/src/nicheformer/data/dataset.py
  references/nicheformer/src/nicheformer/models/_utils.py
"""

import gc

import numba
import numpy as np
import torch
from anndata import AnnData
from scipy.sparse import issparse
from sklearn.utils import sparsefuncs
from torch.utils.data import Dataset
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def sf_normalize(X):
    """Library-size normalize rows to sum to 10,000.

    Handles both sparse (CSR) and dense matrices. Rows that sum to zero are
    left unchanged (no division-by-zero).
    """
    X = X.copy()
    counts = np.array(X.sum(axis=1)).ravel()
    counts += counts == 0.0  # avoid division by zero
    scaling_factor = 10_000.0 / counts

    if issparse(X):
        sparsefuncs.inplace_row_scale(X, scaling_factor)
    else:
        np.multiply(X, scaling_factor.reshape((-1, 1)), out=X)

    return X


# ---------------------------------------------------------------------------
# Tokenization (numba-accelerated)
# ---------------------------------------------------------------------------

@numba.jit(nopython=True, nogil=True)
def _sub_tokenize_data(x: np.ndarray, max_seq_len: int = 4096, aux_tokens: int = 30):
    """Tokenize a dense expression matrix into gene-rank token indices.

    For each cell, non-zero genes are sorted by expression (descending),
    truncated to *max_seq_len*, and offset by *aux_tokens* to leave room for
    special tokens. Remaining positions are zero-padded.

    Returns an int32 array of shape ``(n_cells, max_seq_len)``.
    """
    scores_final = np.empty((x.shape[0], max_seq_len), dtype=np.int32)
    for i in range(x.shape[0]):
        cell = x[i]
        nonzero_mask = np.nonzero(cell)[0]
        sorted_indices = nonzero_mask[np.argsort(-cell[nonzero_mask])][:max_seq_len]
        sorted_indices = sorted_indices + aux_tokens

        scores = np.zeros(max_seq_len, dtype=np.int32)
        scores[: len(sorted_indices)] = sorted_indices.astype(np.int32)
        scores_final[i, :] = scores

    return scores_final


def tokenize_data(x, median_counts_per_gene, max_seq_len: int = 4096):
    """Full tokenization pipeline: normalize, tech-correct, rank-tokenize.

    Parameters
    ----------
    x : array-like
        Dense expression matrix (n_cells, n_genes).
    median_counts_per_gene : np.ndarray
        Per-gene technology median of shape ``(n_genes,)``.
    max_seq_len : int
        Maximum number of gene tokens per cell.

    Returns
    -------
    np.ndarray
        Int32 token matrix of shape ``(n_cells, max_seq_len)``.
    """
    x = np.nan_to_num(x)
    x = sf_normalize(x)
    med = median_counts_per_gene.copy()
    med += med == 0  # avoid division by zero
    x = x / med.reshape((1, -1))
    return _sub_tokenize_data(x, max_seq_len, 30).astype(np.int32)


# ---------------------------------------------------------------------------
# Masking (inference mode — no actual masking)
# ---------------------------------------------------------------------------

def complete_masking(batch, p, n_tokens):
    """Prepare a batch for the Nicheformer encoder.

    At inference time *p* should be ``0.0`` so no tokens are actually masked.
    The function still remaps padding (0 -> 1), builds an attention mask, and
    populates ``batch['masked_indices']``.

    Ported from ``references/nicheformer/src/nicheformer/models/_utils.py``.
    """
    padding_token = 1
    cls_token = 3

    indices = batch["X"]
    # Original padding token is 0; remap to 1
    indices = torch.where(
        indices == 0, torch.tensor(padding_token, device=indices.device), indices
    )
    batch["X"] = indices

    mask = 1 - torch.bernoulli(torch.ones_like(indices, dtype=torch.float), p)
    mask = mask.to(indices.dtype)

    masked_indices = indices * mask
    masked_indices = torch.where(indices != padding_token, masked_indices, indices)
    mask = torch.where(
        indices == padding_token,
        torch.tensor(padding_token, device=mask.device, dtype=mask.dtype),
        mask,
    )

    masked_indices = torch.where(indices != cls_token, masked_indices, indices)
    mask = torch.where(
        indices == cls_token,
        torch.tensor(padding_token, device=mask.device, dtype=mask.dtype),
        mask,
    )

    # 80/10/10 split for masked positions (irrelevant at p=0.0 but kept for
    # compatibility if someone calls with p>0)
    random_tokens = torch.randint(
        10, n_tokens, size=masked_indices.shape, device=masked_indices.device
    )
    random_tokens = random_tokens * torch.bernoulli(
        torch.ones_like(random_tokens, dtype=torch.float) * 0.1
    ).to(torch.int64)
    masked_indices = torch.where(masked_indices == 0, random_tokens, masked_indices)

    same_tokens = indices.clone()
    same_tokens = same_tokens * torch.bernoulli(
        torch.ones_like(same_tokens, dtype=torch.float) * 0.1
    ).to(torch.int64)
    masked_indices = torch.where(masked_indices == 0, same_tokens, masked_indices)

    batch["masked_indices"] = masked_indices
    batch["mask"] = mask
    batch["attention_mask"] = (masked_indices == padding_token).bool()

    return batch


# ---------------------------------------------------------------------------
# Gene alignment
# ---------------------------------------------------------------------------

def align_genes(adata: AnnData, gene_vocab: list[str], technology_mean: np.ndarray):
    """Align input AnnData genes to the Nicheformer gene vocabulary.

    Reindexes *adata* so that its columns match the genes in *gene_vocab*,
    filling missing genes with zeros. Subsets *technology_mean* to the same
    order. Genes in *adata* that are not in the vocabulary are dropped.

    Parameters
    ----------
    adata : AnnData
        Input data (modified in place via subsetting).
    gene_vocab : list[str]
        Ordered gene names from the Nicheformer vocabulary (Ensembl IDs).
    technology_mean : np.ndarray
        Per-gene technology median, aligned to *gene_vocab*.

    Returns
    -------
    adata_aligned : AnnData
        With columns reindexed to match the vocabulary.
    tech_mean_aligned : np.ndarray
        Technology mean subsetted to matched genes.
    """
    vocab_set = set(gene_vocab)
    input_genes = list(adata.var_names)
    shared = [g for g in gene_vocab if g in set(input_genes)]

    n_input = len(input_genes)
    n_shared = len(shared)
    n_oov = n_input - len(set(input_genes) & vocab_set)

    print(f"[nicheformer] Gene alignment: {n_input} input genes, "
          f"{n_shared} matched to vocabulary, {n_oov} OOV (dropped)")

    if n_shared == 0:
        raise ValueError(
            "No input genes match the Nicheformer vocabulary. "
            "Expected Ensembl IDs (e.g. ENSG00000141510). "
            f"Got: {input_genes[:5]}"
        )

    # Build index mapping: vocab position -> shared gene
    vocab_idx = {g: i for i, g in enumerate(gene_vocab)}
    shared_vocab_indices = np.array([vocab_idx[g] for g in shared])

    # Subset adata to shared genes (in vocabulary order)
    adata_aligned = adata[:, shared].copy()
    tech_mean_aligned = technology_mean[shared_vocab_indices]

    return adata_aligned, tech_mean_aligned


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class NicheformerDataset(Dataset):
    """PyTorch Dataset for Nicheformer inference.

    Pre-tokenizes all cells during construction (in chunks to limit memory).
    Each item is a dict with key ``'X'`` containing an int32 token tensor.
    """

    def __init__(self, adata: AnnData, technology_mean: np.ndarray,
                 max_seq_len: int = 1500, chunk_size: int = 1000):
        self.n_cells = adata.n_obs
        self.max_seq_len = max_seq_len
        self.tokens = self._tokenize_chunked(adata, technology_mean, chunk_size)

    def _tokenize_chunked(self, adata, technology_mean, chunk_size):
        n_chunks = (self.n_cells + chunk_size - 1) // chunk_size
        token_chunks = []

        for i in tqdm(range(n_chunks), desc="Tokenizing"):
            start = i * chunk_size
            end = min(start + chunk_size, self.n_cells)
            chunk = adata[start:end]

            x = chunk.X.toarray() if issparse(chunk.X) else np.asarray(chunk.X)
            tokens = tokenize_data(x, technology_mean, self.max_seq_len)
            token_chunks.append(tokens)
            gc.collect()

        return np.concatenate(token_chunks, axis=0)

    def __len__(self):
        return self.n_cells

    def __getitem__(self, idx):
        return {"X": torch.tensor(self.tokens[idx])}
