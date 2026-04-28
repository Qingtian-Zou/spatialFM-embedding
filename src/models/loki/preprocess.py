"""Loki preprocessing helpers (top-50 gene strings + image patch segmentation).

Vendored from references/Loki/src/loki/preprocess.py with one fix in
``segment_patches``. Upstream uses a non-intuitive convention where the
``pixel_x`` / ``pixel_y`` column names are *swapped* relative to their
meanings: its ``load_data_for_annotation`` writes rows into ``pixel_x``
and cols into ``pixel_y``, and ``segment_patches`` then unpacks them with
a matching swap (``ycenter, xcenter = coord[..., ["pixel_x", "pixel_y"]]``).
The two swaps cancel, so the OmiCLIP model was trained on correctly
aligned patches.

Our adapter (``src/adapters/loki.py``) populates the coord DataFrame with
intuitive naming — ``pixel_x`` = x = col, ``pixel_y`` = y = row — which
breaks that cancellation. To restore agreement with the model's training
distribution, ``segment_patches`` here unpacks x first, then y. See the
inline comment in that function.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

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


def _save_patch(img_array, spot_idx, x1, y1, x2, y2, patch_dir):
    patch_img = Image.fromarray(img_array[y1:y2, x1:x2].astype(np.uint8))
    patch_img.save(os.path.join(patch_dir, f"{spot_idx}_hires.png"))


def segment_patches(img_array, coord, patch_dir, height=20, width=20, n_workers: Optional[int] = None):
    """Crop ``height x width`` patches centered at each (pixel_x, pixel_y) coord
    and save as ``<spot_id>_hires.png`` under ``patch_dir``.

    ``img_array`` is expected to be ``uint8`` with values in ``[0, 255]`` — the
    Loki adapter's ``_resolve_spatial`` enforces this contract. Out-of-range
    patches are skipped (no file written).

    Cropping and PNG encoding are parallelized across threads — the dominant
    cost (libpng/zlib + the file ``write()`` syscall) releases the GIL, and
    each spot writes a distinct file, so threads scale near-linearly without
    pickling the image. Pass ``n_workers=1`` to force the sequential path;
    the default (``None``) auto-picks ``min(8, os.cpu_count())``.
    """
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)

    yrange, xrange = img_array.shape[:2]

    tasks = []
    for spot_idx in coord.index:
        # Diverges from upstream (references/Loki/src/loki/preprocess.py:80), which
        # unpacks as ``ycenter, xcenter = coord[..., ["pixel_x", "pixel_y"]]``.
        # That swap is internally consistent with upstream's
        # ``load_data_for_annotation``, where ``pixel_x`` actually stores rows and
        # ``pixel_y`` stores cols — the two swaps cancel, so the model was trained
        # on correctly aligned patches. Our adapter assigns ``pixel_x`` from the
        # column index (x) and ``pixel_y`` from the row index (y), so we keep
        # intuitive naming and unpack x first to match what the model saw at
        # training time.
        xcenter, ycenter = coord.loc[spot_idx, ["pixel_x", "pixel_y"]]

        x1 = round(xcenter - width / 2)
        y1 = round(ycenter - height / 2)
        x2 = x1 + width
        y2 = y1 + height

        if x1 < 0 or y1 < 0 or x2 > xrange or y2 > yrange:
            print(f"Patch {spot_idx} is out of range and will be skipped.")
            continue

        tasks.append((spot_idx, x1, y1, x2, y2))

    if not tasks:
        return

    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 1)
    n_workers = min(n_workers, len(tasks))

    if n_workers <= 1:
        for spot_idx, x1, y1, x2, y2 in tasks:
            _save_patch(img_array, spot_idx, x1, y1, x2, y2, patch_dir)
        return

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        # list() forces consumption so exceptions surface here rather than at
        # GC time, and the executor's __exit__ waits for all writes to finish.
        list(ex.map(
            lambda t: _save_patch(img_array, t[0], t[1], t[2], t[3], t[4], patch_dir),
            tasks,
        ))
