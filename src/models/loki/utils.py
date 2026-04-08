"""OmiCLIP (COCA ViT-L-14) loading and encoding utilities.

Vendored from references/Loki/src/loki/utils.py with unrelated helpers and the
unused cv2 dependency removed.
"""

from typing import List, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image
from open_clip import create_model_from_pretrained, get_tokenizer


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
) -> torch.Tensor:
    """Batch-encode a list of image file paths into L2-normalized embeddings."""
    image_embeddings = []
    for image_path in image_paths:
        image = Image.open(image_path)
        image_input = torch.stack([preprocess(image)]).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        image_embeddings.append(image_features)
    image_embeddings = torch.cat(image_embeddings, dim=0)
    return F.normalize(image_embeddings, p=2, dim=-1)


def encode_texts(
    model: torch.nn.Module,
    tokenizer: callable,
    texts: List[str],
    device: Union[str, torch.device],
) -> torch.Tensor:
    """Batch-encode a list of strings into L2-normalized embeddings."""
    text_inputs = tokenizer(texts).to(device)
    with torch.no_grad():
        feats = model.encode_text(text_inputs)
    return F.normalize(feats, p=2, dim=-1)
