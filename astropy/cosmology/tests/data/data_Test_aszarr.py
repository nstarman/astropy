# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np

from astropy.cosmology.tests.conftest import _zarr, valid_zs

test_results = {
    "zs": {i: z for i, z in enumerate(valid_zs)},
    "valid": {
        0: 0,
        1: 1,
        2: 1100,
        3: np.float64(3300),
        4: 2.0,
        5: 3.0,
        6: _zarr,
        7: _zarr,
        8: _zarr,
        9: _zarr
    }
}
