"""
Gene vocabulary for scGPT-spatial.

Replaces the original torchtext.vocab.Vocab-based GeneVocab with a pure-Python
dict wrapper that preserves the same API surface used in the inference path.
"""

import json
import pickle
from pathlib import Path
from collections import Counter, OrderedDict
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch

from . import logger


class GeneVocab:
    """
    Vocabulary for genes. Drop-in replacement for the original torchtext-based
    GeneVocab, using an internal OrderedDict for token→index mapping.
    """

    def __init__(
        self,
        gene_list_or_vocab: Union[List[str], "GeneVocab"],
        specials: Optional[List[str]] = None,
        special_first: bool = True,
        default_token: Optional[str] = "<pad>",
    ) -> None:
        self._default_index: Optional[int] = None
        self._pad_token: Optional[str] = None

        if isinstance(gene_list_or_vocab, GeneVocab):
            if specials is not None:
                raise ValueError(
                    "receive non-empty specials when init from a GeneVocab object."
                )
            self._token2idx = OrderedDict(gene_list_or_vocab._token2idx)
            self._idx2token = dict(gene_list_or_vocab._idx2token)
            self._default_index = gene_list_or_vocab._default_index
        elif isinstance(gene_list_or_vocab, list):
            built = self._build_vocab_from_iterator(
                gene_list_or_vocab,
                specials=specials,
                special_first=special_first,
            )
            self._token2idx = built._token2idx
            self._idx2token = built._idx2token
            self._default_index = built._default_index
        else:
            raise ValueError(
                "gene_list_or_vocab must be a list of gene names or a GeneVocab object."
            )

        if default_token is not None and default_token in self:
            self.set_default_token(default_token)

    @classmethod
    def _create_empty(cls) -> "GeneVocab":
        """Create an empty GeneVocab without triggering __init__ recursion."""
        obj = object.__new__(cls)
        obj._token2idx = OrderedDict()
        obj._idx2token = {}
        obj._default_index = None
        obj._pad_token = None
        return obj

    # ---- Core dict-like API ----

    def __getitem__(self, token: str) -> int:
        if token in self._token2idx:
            return self._token2idx[token]
        if self._default_index is not None:
            return self._default_index
        raise KeyError(f"Token '{token}' not found in vocabulary and no default index set.")

    def __contains__(self, token: str) -> bool:
        return token in self._token2idx

    def __call__(self, tokens: List[str]) -> List[int]:
        """Batch lookup: convert a list of tokens to their indices."""
        return [self[t] for t in tokens]

    def __len__(self) -> int:
        return len(self._token2idx)

    def __iter__(self):
        return iter(self._token2idx)

    # ---- Index management ----

    def set_default_index(self, idx: int) -> None:
        self._default_index = idx

    def get_default_index(self) -> Optional[int]:
        return self._default_index

    def get_stoi(self) -> Dict[str, int]:
        """Return a copy of the string-to-index mapping."""
        return dict(self._token2idx)

    def get_itos(self) -> List[str]:
        """Return index-to-string list."""
        return [self._idx2token[i] for i in range(len(self._idx2token))]

    # ---- Token insertion ----

    def insert_token(self, token: str, idx: int) -> None:
        """Insert a token at a specific index. Assumes sequential insertion."""
        if token in self._token2idx:
            return
        self._token2idx[token] = idx
        self._idx2token[idx] = token

    def append_token(self, token: str) -> None:
        """Append a token at the end of the vocabulary."""
        if token in self._token2idx:
            return
        idx = len(self._token2idx)
        self._token2idx[token] = idx
        self._idx2token[idx] = token

    # ---- Pad token property ----

    @property
    def pad_token(self) -> Optional[str]:
        return self._pad_token

    @pad_token.setter
    def pad_token(self, pad_token: str) -> None:
        if pad_token not in self:
            raise ValueError(f"{pad_token} is not in the vocabulary.")
        self._pad_token = pad_token

    def set_default_token(self, default_token: str) -> None:
        if default_token not in self:
            raise ValueError(f"{default_token} is not in the vocabulary.")
        self.set_default_index(self[default_token])

    # ---- Serialization ----

    @classmethod
    def from_file(cls, file_path: Union[Path, str]) -> "GeneVocab":
        if isinstance(file_path, str):
            file_path = Path(file_path)
        if file_path.suffix == ".pkl":
            with file_path.open("rb") as f:
                vocab = pickle.load(f)
                return cls(vocab)
        elif file_path.suffix == ".json":
            with file_path.open("r") as f:
                token2idx = json.load(f)
                return cls.from_dict(token2idx)
        else:
            raise ValueError(
                f"{file_path} is not a valid file type. "
                "Only .pkl and .json are supported."
            )

    @classmethod
    def from_dict(
        cls,
        token2idx: Dict[str, int],
        default_token: Optional[str] = "<pad>",
    ) -> "GeneVocab":
        _vocab = cls._create_empty()
        for t, i in sorted(token2idx.items(), key=lambda x: x[1]):
            _vocab.insert_token(t, i)
        if default_token is not None and default_token in _vocab:
            _vocab.set_default_token(default_token)
        return _vocab

    def save_json(self, file_path: Union[Path, str]) -> None:
        if isinstance(file_path, str):
            file_path = Path(file_path)
        with file_path.open("w") as f:
            json.dump(self.get_stoi(), f, indent=2)

    # ---- Build from iterator ----

    @classmethod
    def _build_vocab_from_iterator(
        cls,
        iterator: Iterable,
        min_freq: int = 1,
        specials: Optional[List[str]] = None,
        special_first: bool = True,
    ) -> "GeneVocab":
        counter = Counter()
        counter.update(iterator)

        if specials is not None:
            for tok in specials:
                if tok in counter:
                    del counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials_to_add = specials[::-1]
            else:
                specials_to_add = specials
            for symbol in specials_to_add:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)

        vocab = cls._create_empty()
        for idx, token in enumerate(ordered_dict):
            vocab._token2idx[token] = idx
            vocab._idx2token[idx] = token

        return vocab


# ---------------------------------------------------------------------------
# Tokenization utilities (unchanged from original, only type hints updated)
# ---------------------------------------------------------------------------

def tokenize_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    return_pt: bool = True,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_id: int = "<cls>",
    mod_type: np.ndarray = None,
    cls_id_mod_type: int = None,
    sample_zero: bool = False,
) -> list:
    if data.shape[1] != len(gene_ids):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of gene_ids ({len(gene_ids)})."
        )
    if mod_type is not None and data.shape[1] != len(mod_type):
        raise ValueError(
            f"Number of features in data ({data.shape[1]}) does not match "
            f"number of mod_type ({len(mod_type)})."
        )

    tokenized_data = []
    for i in range(len(data)):
        row = data[i]
        mod_types = None
        if include_zero_gene:
            values = row
            genes = gene_ids
            if mod_type is not None:
                mod_types = mod_type
        elif sample_zero:
            values = row
            idx_nonzero = np.where(values != 0)[0]
            idx_zero = np.where(values == 0)[0]
            len_subset_zero = min(int(len(idx_nonzero) * 0.4), len(idx_zero))
            subset_idx_zero = np.random.choice(len(idx_zero), len_subset_zero, replace=False)
            idx_zero = idx_zero[subset_idx_zero]
            gene_ids_zero = gene_ids[idx_zero]
            values_zero = values[idx_zero]
            genes = np.concatenate((gene_ids[idx_nonzero], gene_ids_zero), axis=0)
            values = np.concatenate((values[idx_nonzero], values_zero), axis=0)
            if mod_type is not None:
                mod_types = mod_type
        else:
            idx = np.nonzero(row)[0]
            values = row[idx]
            genes = gene_ids[idx]
            if mod_type is not None:
                mod_types = mod_type[idx]

        if append_cls:
            genes = np.insert(genes, 0, cls_id)
            values = np.insert(values, 0, 0)
            if mod_type is not None:
                mod_types = np.insert(mod_types, 0, cls_id_mod_type)
        if return_pt:
            genes = torch.from_numpy(genes).long()
            values = torch.from_numpy(values).float()
            if mod_type is not None:
                mod_types = torch.from_numpy(mod_types).long()
        tokenized_data.append((genes, values, mod_types))
    return tokenized_data


def pad_batch(
    batch: list,
    max_len: int,
    vocab,
    pad_token: str = "<pad>",
    pad_value: int = 0,
    cls_appended: bool = True,
    vocab_mod=None,
) -> Dict[str, torch.Tensor]:
    max_ori_len = max(len(batch[i][0]) for i in range(len(batch)))
    max_len = min(max_ori_len, max_len)

    pad_id = vocab[pad_token]
    if vocab_mod is not None:
        mod_pad_id = vocab_mod[pad_token]
    gene_ids_list = []
    values_list = []
    mod_types_list = []

    for i in range(len(batch)):
        gene_ids, values, mod_types = batch[i]

        if len(gene_ids) > max_len:
            if not cls_appended:
                idx = np.random.choice(len(gene_ids), max_len, replace=False)
            else:
                idx = np.random.choice(len(gene_ids) - 1, max_len - 1, replace=False)
                idx = idx + 1
                idx = np.insert(idx, 0, 0)
            gene_ids = gene_ids[idx]
            values = values[idx]
            if mod_types is not None:
                mod_types = mod_types[idx]

        if len(gene_ids) < max_len:
            gene_ids = torch.cat(
                [
                    gene_ids,
                    torch.full(
                        (max_len - len(gene_ids),), pad_id, dtype=gene_ids.dtype
                    ),
                ]
            )
            values = torch.cat(
                [
                    values,
                    torch.full((max_len - len(values),), pad_value, dtype=values.dtype),
                ]
            )
            if mod_types is not None:
                mod_types = torch.cat(
                    [
                        mod_types,
                        torch.full(
                            (max_len - len(mod_types),),
                            mod_pad_id,
                            dtype=mod_types.dtype,
                        ),
                    ]
                )

        gene_ids_list.append(gene_ids)
        values_list.append(values)
        if mod_types is not None:
            mod_types_list.append(mod_types)

    batch_padded = {
        "genes": torch.stack(gene_ids_list, dim=0),
        "values": torch.stack(values_list, dim=0),
    }
    if mod_types is not None:
        batch_padded["mod_types"] = torch.stack(mod_types_list, dim=0)
    return batch_padded


def tokenize_and_pad_batch(
    data: np.ndarray,
    gene_ids: np.ndarray,
    max_len: int,
    vocab,
    pad_token: str,
    pad_value: int,
    append_cls: bool = True,
    include_zero_gene: bool = False,
    cls_token: str = "<cls>",
    return_pt: bool = True,
    mod_type: np.ndarray = None,
    vocab_mod=None,
    sample_zero: bool = False,
) -> Dict[str, torch.Tensor]:
    cls_id = vocab[cls_token]
    if mod_type is not None:
        cls_id_mod_type = vocab_mod[cls_token]
    tokenized_data = tokenize_batch(
        data,
        gene_ids,
        return_pt=return_pt,
        append_cls=append_cls,
        include_zero_gene=include_zero_gene,
        cls_id=cls_id,
        mod_type=mod_type,
        cls_id_mod_type=cls_id_mod_type if mod_type is not None else None,
        sample_zero=sample_zero,
    )

    batch_padded = pad_batch(
        tokenized_data,
        max_len,
        vocab,
        pad_token,
        pad_value,
        cls_appended=append_cls,
        vocab_mod=vocab_mod,
    )
    return batch_padded


def random_mask_value(
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    pad_value: int = 0,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    for i in range(len(values)):
        row = values[i]
        non_padding_idx = np.nonzero(row - pad_value)[0]
        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)
        row[mask_idx] = mask_value
    return torch.from_numpy(values).float()


def random_mask_gene_value(
    genes: Union[torch.Tensor, np.ndarray],
    values: Union[torch.Tensor, np.ndarray],
    mask_ratio: float = 0.15,
    mask_value: int = -1,
    mask_gene: int = None,
    pad_value: int = 0,
    mask_gene_ratio: int = 0.5,
    sample_zero: bool = False,
) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        values = values.clone().detach().numpy()
    else:
        values = values.copy()

    if isinstance(genes, torch.Tensor):
        genes = genes.clone().detach().numpy()
    else:
        genes = genes.copy()

    assert len(genes) == len(values)

    for i in range(len(genes)):
        gene_row = genes[i]
        row = values[i]

        non_padding_idx = np.nonzero(row - pad_value)[0]

        n_mask = int(len(non_padding_idx) * mask_ratio)
        mask_idx = np.random.choice(non_padding_idx, n_mask, replace=False)

        if sample_zero:
            non_zero_padding_idx = np.where((row != 0) & (row != pad_value))[0]
            n_mask = int(len(non_zero_padding_idx) * mask_ratio)
            mask_idx_non_zero = np.random.choice(non_zero_padding_idx, n_mask, replace=False)
            mask_idx_zero = np.where(row == 0)[0]
            mask_idx = np.concatenate((mask_idx_non_zero, mask_idx_zero), axis=0)

        if mask_gene_ratio > 0:
            mask_gene_idx = mask_idx[: int(n_mask * mask_gene_ratio)]
            mask_val_idx = mask_idx[int(n_mask * mask_gene_ratio) :]
            gene_row[mask_gene_idx] = mask_gene
            row[mask_val_idx] = mask_value
        else:
            row[mask_idx] = mask_value
    return torch.from_numpy(genes).long(), torch.from_numpy(values).float()
