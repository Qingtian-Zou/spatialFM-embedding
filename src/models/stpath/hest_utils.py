"""Narrow vendor of stpath/hest_utils/file_utils.py — only the H5 reader."""

import h5py


def read_assets_from_h5(h5_path, keys=None, skip_attrs=False, skip_assets=False):
    """Read all (or specified) datasets from an HDF5 file.

    Returns: (assets dict, attrs dict).
    """
    assets = {}
    attrs = {}
    with h5py.File(h5_path, 'r') as f:
        if keys is None:
            keys = list(f.keys())
        for key in keys:
            if not skip_assets:
                assets[key] = f[key][:]
            if not skip_attrs:
                if f[key].attrs is not None:
                    attrs[key] = dict(f[key].attrs)
    return assets, attrs
