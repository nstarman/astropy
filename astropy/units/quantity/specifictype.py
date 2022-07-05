# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .core import Quantity

if TYPE_CHECKING:
    from astropy.units.core import UnitBase

    # from typing_extensions import Self  # noqa: F401
    Self = Any


__all__ = ["SpecificTypeQuantity"]


class SpecificTypeQuantity(Quantity):
    """Superclass for Quantities of specific physical type.

    Subclasses of these work just like :class:`~astropy.units.Quantity`, except
    that they are for specific physical types (and may have methods that are
    only appropriate for that type).  Astropy examples are
    :class:`~astropy.coordinates.Angle` and
    :class:`~astropy.coordinates.Distance`

    At a minimum, subclasses should set ``_equivalent_unit`` to the unit
    associated with the physical type.
    """
    # The unit for the specific physical type.  Instances can only be created
    # with units that are equivalent to this.
    _equivalent_unit = None

    # The default unit used for views.  Even with `None`, views of arrays
    # without units are possible, but will have an uninitialized unit.
    _unit = None

    # Default unit for initialization through the constructor.
    _default_unit = None

    # ensure that we get precedence over our superclass.
    __array_priority__ = Quantity.__array_priority__ + 10

    def __quantity_subclass__(self, unit: UnitBase):
        if unit.is_equivalent(self._equivalent_unit):
            return type(self), True
        else:
            return super().__quantity_subclass__(unit)[0], False

    def _set_unit(self, unit):
        if unit is None or not unit.is_equivalent(self._equivalent_unit):
            raise UnitTypeError(
                "{} instances require units equivalent to '{}'"
                .format(type(self).__name__, self._equivalent_unit) +
                (", but no unit was given." if unit is None else
                 f", so cannot set it to '{unit}'."))

        super()._set_unit(unit)
