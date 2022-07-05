# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

from __future__ import annotations

# STDLIB
from abc import ABCMeta
from fractions import Fraction
from typing import Any, ClassVar, Dict

# THIRD PARTY
import numpy as np

from astropy.units.core import Unit

# LOCAL
from .facets import BaseQuantity, NDArrayQuantity

__all__ = ["Quantity"]


# We don't want to run doctests in the docstrings we inherit from Numpy
__doctest_skip__ = ['Quantity.*']


_UFUNCS_FILTER_WARNINGS = {np.arcsin, np.arccos, np.arccosh, np.arctanh}


class QuantityMeta(ABCMeta):

    _router: ClassVar[Dict[type, type]] = {}
    # TODO! use entry points for adding to the router

    def __call__(self, value, *args: Any, **kwds: Any) -> Any:

        # Dispatch elsewhere
        if self is Quantity:
            for base, qcls in self._router.items():
                if isinstance(value, base):
                    return qcls(value, *args, **kwds)

        return super().__call__(value, *args, **kwds)


class Quantity(NDArrayQuantity, BaseQuantity, metaclass=QuantityMeta):
    """A `~astropy.units.Quantity` represents a number with some associated unit.

    See also: https://docs.astropy.org/en/stable/units/quantity.html

    Parameters
    ----------
    value : number, `~numpy.ndarray`, `~astropy.units.Quantity` (sequence), or str
        The numerical value of this quantity in the units given by unit.  If a
        `Quantity` or sequence of them (or any other valid object with a
        ``unit`` attribute), creates a new `Quantity` object, converting to
        `unit` units as needed.  If a string, it is converted to a number or
        `Quantity`, depending on whether a unit is present.

    unit : unit-like
        An object that represents the unit associated with the input value.
        Must be an `~astropy.units.UnitBase` object or a string parseable by
        the :mod:`~astropy.units` package.

    dtype : ~numpy.dtype, optional
        The dtype of the resulting Numpy array or scalar that will
        hold the value.  If not provided, it is determined from the input,
        except that any integer and (non-Quantity) object inputs are converted
        to float by default.
        If `None`, the normal `numpy.dtype` introspection is used, e.g.
        preventing upcasting of integers.

    copy : bool, optional
        If `True` (default), then the value is copied.  Otherwise, a copy will
        only be made if ``__array__`` returns a copy, if value is a nested
        sequence, or if a copy is needed to satisfy an explicitly given
        ``dtype``.  (The `False` option is intended mostly for internal use,
        to speed up initialization where a copy is known to have been made.
        Use with care.)

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  As in `~numpy.array`.  This parameter
        is ignored if the input is a `Quantity` and ``copy=False``.

    subok : bool, optional
        If `False` (default), the returned array will be forced to be a
        `Quantity`.  Otherwise, `Quantity` subclasses will be passed through,
        or a subclass appropriate for the unit will be used (such as
        `~astropy.units.Dex` for ``u.dex(u.AA)``).

    ndmin : int, optional
        Specifies the minimum number of dimensions that the resulting array
        should have.  Ones will be pre-pended to the shape as needed to meet
        this requirement.  This parameter is ignored if the input is a
        `Quantity` and ``copy=False``.

    Raises
    ------
    TypeError
        If the value provided is not a Python numeric type.
    TypeError
        If the unit provided is not either a :class:`~astropy.units.Unit`
        object or a parseable string unit.

    Notes
    -----
    Quantities can also be created by multiplying a number or array with a
    :class:`~astropy.units.Unit`. See https://docs.astropy.org/en/latest/units/

    Unless the ``dtype`` argument is explicitly specified, integer
    or (non-Quantity) object inputs are converted to `float` by default.
    """

    def __quantity_subclass__(self, unit):
        """
        Overridden by subclasses to change what kind of view is
        created based on the output unit of an operation.

        Parameters
        ----------
        unit : UnitBase
            The unit for which the appropriate class should be returned

        Returns
        -------
        tuple :
            - `~astropy.units.Quantity` subclass
            - bool: True if subclasses of the given class are ok
        """
        return Quantity, True

    # TODO! move to NDArrayQuantity
    def _new_view(self, obj=None, unit=None):
        """
        Create a Quantity view of some array-like input, and set the unit

        By default, return a view of ``obj`` of the same class as ``self`` and
        with the same unit.  Subclasses can override the type of class for a
        given unit using ``__quantity_subclass__``, and can ensure properties
        other than the unit are copied using ``__array_finalize__``.

        If the given unit defines a ``_quantity_class`` of which ``self``
        is not an instance, a view using this class is taken.

        Parameters
        ----------
        obj : ndarray or scalar, optional
            The array to create a view of.  If obj is a numpy or python scalar,
            it will be converted to an array scalar.  By default, ``self``
            is converted.

        unit : unit-like, optional
            The unit of the resulting object.  It is used to select a
            subclass, and explicitly assigned to the view if given.
            If not given, the subclass and unit will be that of ``self``.

        Returns
        -------
        view : `~astropy.units.Quantity` subclass
        """
        # Determine the unit and quantity subclass that we need for the view.
        if unit is None:
            unit = self.unit
            quantity_subclass = self.__class__
        elif unit is self.unit and self.__class__ is Quantity:
            # The second part is because we should not presume what other
            # classes want to do for the same unit.  E.g., Constant will
            # always want to fall back to Quantity, and relies on going
            # through `__quantity_subclass__`.
            quantity_subclass = Quantity
        else:
            unit = Unit(unit)
            quantity_subclass = getattr(unit, '_quantity_class', Quantity)
            if isinstance(self, quantity_subclass):
                quantity_subclass, subok = self.__quantity_subclass__(unit)
                if subok:
                    quantity_subclass = self.__class__

        # We only want to propagate information from ``self`` to our new view,
        # so obj should be a regular array.  By using ``np.array``, we also
        # convert python and numpy scalars, which cannot be viewed as arrays
        # and thus not as Quantity either, to zero-dimensional arrays.
        # (These are turned back into scalar in `.value`)
        # Note that for an ndarray input, the np.array call takes only double
        # ``obj.__class is np.ndarray``. So, not worth special-casing.
        if obj is None:
            obj = self.view(np.ndarray)
        else:
            obj = np.array(obj, copy=False, subok=True)

        # Take the view, set the unit, and update possible other properties
        # such as ``info``, ``wrap_angle`` in `Longitude`, etc.
        view = obj.view(quantity_subclass)
        view._set_unit(unit)
        view.__array_finalize__(self)
        return view

    def __pow__(self, other):
        if isinstance(other, Fraction):
            # Avoid getting object arrays by raising the value to a Fraction.
            return self._new_view(self.value ** float(other),
                                self.unit ** other)

        return super().__pow__(other)
