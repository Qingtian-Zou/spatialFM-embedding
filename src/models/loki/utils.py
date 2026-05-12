"""OmiCLIP (COCA ViT-L-14) loading and encoding utilities.

Vendored from references/Loki/src/loki/utils.py with unrelated helpers and the
unused cv2 dependency removed.
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer
from tqdm import tqdm


def load_model(
    model_path: str,
    device: Union[str, torch.device],
) -> Tuple[torch.nn.Module, callable, callable]:
    """Load pretrained OmiCLIP (COCA ViT-L-14) model, image preprocess, and tokenizer."""
    model, preprocess = create_model_from_pretrained(
        "coca_ViT-L-14", device=device, pretrained=model_path, weights_only=False
    )
    tokenizer = get_tokenizer("coca_ViT-L-14")
    model.to(device).eval()
    return model, preprocess, tokenizer


def encode_images(
    model: torch.nn.Module,
    preprocess: callable,
    image_paths: List[str],
    device: Union[str, torch.device],
    batch_size: int = 64,
) -> torch.Tensor:
    """Batch-encode a list of image file paths into L2-normalized embeddings."""
    image_embeddings = []
    for start in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images", ascii=True):
        chunk = image_paths[start : start + batch_size]
        batch = torch.stack([preprocess(Image.open(p)) for p in chunk]).to(device)
        with torch.no_grad():
            feats = model.encode_image(batch)
        image_embeddings.append(feats)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    return F.normalize(image_embeddings, p=2, dim=-1)


def encode_texts(
    model: torch.nn.Module,
    tokenizer: callable,
    texts: List[str],
    device: Union[str, torch.device],
    batch_size: int = 64,
) -> torch.Tensor:
    """Batch-encode a list of strings into L2-normalized embeddings."""
    text_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Encoding texts", ascii=True):
        chunk = texts[start : start + batch_size]
        text_inputs = tokenizer(chunk).to(device)
        with torch.no_grad():
            feats = model.encode_text(text_inputs)
        text_embeddings.append(feats)
    text_embeddings = torch.cat(text_embeddings, dim=0)
    return F.normalize(text_embeddings, p=2, dim=-1)
