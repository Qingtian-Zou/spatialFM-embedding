"""Loki preprocessing helpers (top-50 gene strings + image patch segmentation).

Vendored from references/Loki/src/loki/preprocess.py with one bug fix in
``segment_patches``: upstream swapped x/y when unpacking the per-spot
coordinates, causing patches to be cropped at the wrong pixel. See the
inline comment in that function.
"""

import os

import numpy as np
import pandas as pd
from PIL import Image


def generate_gene_df(ad, house_keeping_genes, todense=True):
    """Generate a DataFrame with the top-50 expressed genes per spot, as a
    space-separated 'label' string.

    Drops genes containing '.' or '-' and any genes listed in
    ``house_keeping_genes['genesymbol']``.
    """
    ad = ad[:, ~ad.var.index.str.contains('.', regex=False)]
    ad = ad[:, ~ad.var.index.str.contains('-', regex=False)]
    ad = ad[:, ~ad.var.index.isin(house_keeping_genes['genesymbol'])]

    if todense:
        expr = pd.DataFrame(ad.X.todense(), index=ad.obs.index, columns=ad.var.index)
    else:
        expr = pd.DataFrame(ad.X, index=ad.obs.index, columns=ad.var.index)

    top_k_genes = expr.apply(lambda s, n: pd.Series(s.nlargest(n).index), axis=1, n=50)

    top_k_genes_str = pd.DataFrame()
    top_k_genes_str['label'] = top_k_genes[top_k_genes.columns].astype(str) \
        .apply(lambda x: ' '.join(x), axis=1)
    return top_k_genes_str


def segment_patches(img_array, coord, patch_dir, height=20, width=20):
    """Crop ``height x width`` patches centered at each (pixel_x, pixel_y) coord
    and save as ``<spot_id>_hires.png`` under ``patch_dir``.

    ``img_array`` is expected to be ``uint8`` with values in ``[0, 255]`` — the
    Loki adapter's ``_resolve_spatial`` enforces this contract. Out-of-range
    patches are skipped (no file written).
    """
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    yrange, xrange = img_array.shape[:2]

    for spot_idx in coord.index:
        # Diverges from upstream (references/Loki/src/loki/preprocess.py): upstream
        # unpacks as ``ycenter, xcenter = coord[..., ["pixel_x", "pixel_y"]]``,
        # which swaps the axes. Our adapter sets ``pixel_x`` from the column index
        # (x-axis) and ``pixel_y`` from the row index (y-axis), so the correct
        # unpacking is x first, then y.
        xcenter, ycenter = coord.loc[spot_idx, ["pixel_x", "pixel_y"]]

        x1 = round(xcenter - width / 2)
        y1 = round(ycenter - height / 2)
        x2 = x1 + width
        y2 = y1 + height

        if x1 < 0 or y1 < 0 or x2 > xrange or y2 > yrange:
            print(f"Patch {spot_idx} is out of range and will be skipped.")
            continue

        patch_img = Image.fromarray(img_array[y1:y2, x1:x2].astype(np.uint8))
        patch_img.save(os.path.join(patch_dir, f"{spot_idx}_hires.png"))
