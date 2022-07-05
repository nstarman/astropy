# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

from __future__ import annotations

# THIRD PARTY
import numpy as np

# LOCAL
from astropy.units.core import UnitsError, dimensionless_unscaled
from astropy.units.quantity.core import Quantity

__all__ = ["allclose", "isclose"]


def isclose(a, b, rtol=1.e-5, atol=None, equal_nan=False, **kwargs):
    """
    Return a boolean array where two arrays are element-wise equal
    within a tolerance.

    Parameters
    ----------
    a, b : array-like or `~astropy.units.Quantity`
        Input values or arrays to compare
    rtol : array-like or `~astropy.units.Quantity`
        The relative tolerance for the comparison, which defaults to
        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,
        then it must be dimensionless.
    atol : number or `~astropy.units.Quantity`
        The absolute tolerance for the comparison.  The units (or lack
        thereof) of ``a``, ``b``, and ``atol`` must be consistent with
        each other.  If `None`, ``atol`` defaults to zero in the
        appropriate units.
    equal_nan : `bool`
        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will
        be considered equal to NaN’s in ``b``.

    Notes
    -----
    This is a :class:`~astropy.units.Quantity`-aware version of
    :func:`numpy.isclose`. However, this differs from the `numpy` function in
    that the default for the absolute tolerance here is zero instead of
    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default
    *absolute* tolerance given two inputs that may have differently scaled
    units.

    Raises
    ------
    `~astropy.units.UnitsError`
        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,
        or if ``rtol`` is not dimensionless.

    See also
    --------
    allclose
    """
    unquantified_args = _unquantify_allclose_arguments(a, b, rtol, atol)
    return np.isclose(*unquantified_args, equal_nan=equal_nan, **kwargs)


def allclose(a, b, rtol=1.e-5, atol=None, equal_nan=False, **kwargs) -> bool:
    """
    Whether two arrays are element-wise equal within a tolerance.

    Parameters
    ----------
    a, b : array-like or `~astropy.units.Quantity`
        Input values or arrays to compare
    rtol : array-like or `~astropy.units.Quantity`
        The relative tolerance for the comparison, which defaults to
        ``1e-5``.  If ``rtol`` is a :class:`~astropy.units.Quantity`,
        then it must be dimensionless.
    atol : number or `~astropy.units.Quantity`
        The absolute tolerance for the comparison.  The units (or lack
        thereof) of ``a``, ``b``, and ``atol`` must be consistent with
        each other.  If `None`, ``atol`` defaults to zero in the
        appropriate units.
    equal_nan : `bool`
        Whether to compare NaN’s as equal. If `True`, NaNs in ``a`` will
        be considered equal to NaN’s in ``b``.

    Notes
    -----
    This is a :class:`~astropy.units.Quantity`-aware version of
    :func:`numpy.allclose`. However, this differs from the `numpy` function in
    that the default for the absolute tolerance here is zero instead of
    ``atol=1e-8`` in `numpy`, as there is no natural way to set a default
    *absolute* tolerance given two inputs that may have differently scaled
    units.

    Raises
    ------
    `~astropy.units.UnitsError`
        If the dimensions of ``a``, ``b``, or ``atol`` are incompatible,
        or if ``rtol`` is not dimensionless.

    See also
    --------
    isclose
    """
    unquantified_args = _unquantify_allclose_arguments(a, b, rtol, atol)
    return np.allclose(*unquantified_args, equal_nan=equal_nan, **kwargs)


def _unquantify_allclose_arguments(actual, desired, rtol, atol):
    actual = Quantity(actual, subok=True, copy=False)

    desired = Quantity(desired, subok=True, copy=False)
    try:
        desired = desired.to(actual.unit)
    except UnitsError:
        raise UnitsError(
            f"Units for 'desired' ({desired.unit}) and 'actual' "
            f"({actual.unit}) are not convertible"
        )

    if atol is None:
        # By default, we assume an absolute tolerance of zero in the
        # appropriate units.  The default value of None for atol is
        # needed because the units of atol must be consistent with the
        # units for a and b.
        atol = Quantity(0)
    else:
        atol = Quantity(atol, subok=True, copy=False)
        try:
            atol = atol.to(actual.unit)
        except UnitsError:
            raise UnitsError(
                f"Units for 'atol' ({atol.unit}) and 'actual' "
                f"({actual.unit}) are not convertible"
            )

    rtol = Quantity(rtol, subok=True, copy=False)
    try:
        rtol = rtol.to(dimensionless_unscaled)
    except Exception:
        raise UnitsError("'rtol' should be dimensionless")

    return actual.value, desired.value, rtol.value, atol.value
