# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The following are private functions. These functions are registered into
:meth:`~astropy.cosmology.Cosmology.to_format` and
:meth:`~astropy.cosmology.Cosmology.from_format` and should only be accessed
via these methods.
"""

from collections import OrderedDict

import numpy as np

from astropy.cosmology.core import _COSMOLOGY_CLASSES, Cosmology
from astropy.cosmology.connect import convert_registry

from .mapping import from_mapping, to_mapping


__all__ = []  # nothing is publicly scoped


def from_ndarray(array, *, cosmology=None):
    params = {k: array[k] for k in array.dtype.names}

    meta = dict(array.dtype.metadata)
    params["cosmology"] = meta.pop("cosmology")
    params["name"] = meta.pop("name")

    params["meta"] = meta

    return from_mapping(params, move_to_meta=False, cosmology=cosmology)


def to_ndarray(cosmo, *args, cls=np.ndarray):
    """Return the cosmology parameters, and metadata as a `numpy.ndarray`.

    Parameters
    ----------
    cosmo : `~astropy.cosmology.Cosmology`
        The cosmology to return.
    *args
        Not used.
    cls : type (optional, keyword-only)
        Anything compatible with :meth:`numpy.ndarray.view`.
        The array type to return. Default is `numpy.ndarray`.

    Returns
    -------
    ndarray
        Structured.
    """
    m = to_mapping(cosmo, cosmology_as_str=True, cls=OrderedDict)

    # Move from mapping and add to metadata
    meta = m.pop("meta")
    meta["cosmology"] = m.pop("cosmology")
    meta["name"] = m.pop("name")
    meta.move_to_end("name", last=False)  # move to start
    meta.move_to_end("cosmology", last=False)  # move to start

    # Make dtype  # TODO! use the Parameter dtype info when PR merged
    dtype = np.dtype([(k, getattr(v, "dtype", float)) for k, v in m.items()],
                     metadata=meta)

    # Make array
    data = tuple(getattr(v, "value", v) for v in m.values())
    array = np.array(data, dtype=dtype)

    return array.view(cls)


def ndarray_identify(origin, format, *args, **kwargs):
    """Identify if object is a `~astropy.cosmology.Cosmology`.

    Returns
    -------
    bool
    """
    itis = False
    if origin == "read":
        itis = isinstance(args[1], np.ndarray) and (format in (None, "numpy.ndarray"))
    return itis


# ===================================================================
# Register

convert_registry.register_reader("numpy.ndarray", Cosmology, from_ndarray)
convert_registry.register_writer("numpy.ndarray", Cosmology, to_ndarray)
convert_registry.register_identifier("numpy.ndarray", Cosmology, ndarray_identify)
