# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
The following are private functions. These functions are registered into
:meth:`~astropy.cosmology.Cosmology.to_format` and
:meth:`~astropy.cosmology.Cosmology.from_format` and should only be accessed
via these methods.
"""

from collections import OrderedDict

import numpy as np

import astropy.units as u
from astropy.cosmology.core import _COSMOLOGY_CLASSES, Cosmology
from astropy.cosmology.connect import convert_registry

from .ndarray import from_ndarray, to_ndarray


__all__ = []  # nothing is publicly scoped


def from_quantity(quantity, *, cosmology=None):
    return from_ndarray(quantity, cosmology=cosmology)


def to_quantity(cosmo, *args):
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
    array = to_ndarray(cosmo)

    unit = u.Unit(
        tuple(
            getattr(getattr(cosmo, p), "unit", u.one) for p in array.dtype.names
        )
    )
    return array << unit


def quantity_identify(origin, format, *args, **kwargs):
    """Identify if object is a `~astropy.cosmology.Cosmology`.

    Returns
    -------
    bool
    """
    itis = False
    if origin == "read":
        itis = isinstance(args[1], u.Quantity) and (format in (None, "astropy.quantity"))
    return itis


# ===================================================================
# Register

convert_registry.register_reader("astropy.quantity", Cosmology, from_quantity, priority=1)
convert_registry.register_writer("astropy.quantity", Cosmology, to_quantity, priority=1)
convert_registry.register_identifier("astropy.quantity", Cosmology, quantity_identify)
