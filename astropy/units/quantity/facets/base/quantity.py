from __future__ import annotations

import copy
import warnings
from typing import TYPE_CHECKING, Any

from astropy.units.core import (
    Unit, UnitBase, UnitsError, UnitTypeError, dimensionless_unscaled, get_current_unit_registry)
from astropy.units.quantity.base import QuantityBase, _ValueType
from astropy.units.structured import StructuredUnit
from astropy.utils.compat.misc import override__dir__
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyWarning

from .info import QuantityInfo

if TYPE_CHECKING:
    from astropy.units.equivalencies import Equivalency

__all__ = ["BaseQuantity"]

Self = Any  # TODO! have typing_extensions


class BaseQuantity(QuantityBase[_ValueType]):

    _magnitude: _ValueType  # wanted _value, but incompatible with Constant

    _unit: UnitBase

    # Need to set a class-level default for _equivalencies, or
    # Constants can not initialize properly
    _equivalencies: list[Equivalency] = []

    # Default unit for initialization; can be overridden by subclasses,
    # possibly to `None` to indicate there is no default unit.
    _default_unit = dimensionless_unscaled

    info = QuantityInfo()

    def __init__(self, value, unit) -> None:
        self._magnitude = value
        self._unit = Unit(unit)

    # ===============================================================
    # DuckQuantity

    @property
    def value(self) -> _ValueType:
        """The numerical value of this instance.

        See also
        --------
        to_value : Get the numerical value in a given unit.
        """
        return self._magnitude

    @property
    def unit(self):
        """`~astropy.units.UnitBase` representing the unit of this quantity.
        """
        return self._unit

    @property
    def equivalencies(self):
        """
        A list of equivalencies that will be applied by default during
        unit conversions.
        """
        return self._equivalencies

    def to(self: Self, unit: UnitBase | str, equivalencies: list[Equivalency] = [], copy: bool=True) -> Self:
        # make Unit
        unit = Unit(unit)

        # TODO! support copy = False
        value = self.unit.to(unit, self._magnitude, equivalencies=equivalencies)

        return self.__class__(value, unit)

    def to_value(self: Self, unit: UnitBase | str, equivalencies: list[Equivalency] = []) -> _ValueType:
        # TODO! support not copying
        return self.to(unit, equivalencies=equivalencies).value

    # ===============================================================

    def _set_unit(self, unit: UnitBase | str) -> None:
        """Set the unit.

        This is used anywhere the unit is set or modified, i.e., in the
        initilizer, in ``__imul__`` and ``__itruediv__`` for in-place
        multiplication and division by another unit, as well as in
        ``__array_finalize__`` for wrapping up views.  For Quantity, it just
        sets the unit, but subclasses can override it to check that, e.g.,
        a unit is consistent.
        """
        if not isinstance(unit, UnitBase):
            # TODO! support StructuredUnit for this supporting structured dtypes
            # Trying to go through a string ensures that, e.g., Magnitudes with
            # dimensionless physical unit become Quantity with units of mag.
            unit = Unit(str(unit), parse_strict='silent')
            if not isinstance(unit, (UnitBase, StructuredUnit)):
                raise UnitTypeError(
                    "{} instances require normal units, not {} instances."
                    .format(type(self).__name__, type(unit)))

        self._unit = unit

    # This flag controls whether convenience conversion members, such
    # as `q.m` equivalent to `q.to_value(u.m)` are available.  This is
    # not turned on on Quantity itself, but is on some subclasses of
    # Quantity, such as `astropy.coordinates.Angle`.
    _include_easy_conversion_members: bool = False

    @override__dir__
    def __dir__(self):
        """
        Quantities are able to directly convert to other units that
        have the same physical type.  This function is implemented in
        order to make autocompletion still work correctly in IPython.
        """
        if not self._include_easy_conversion_members:
            return []
        extra_members = set()
        equivalencies = Unit._normalize_equivalencies(self.equivalencies)
        for equivalent in self.unit._get_units_with_same_physical_type(
                equivalencies):
            extra_members.update(equivalent.names)
        return extra_members

    def __getattr__(self, attr):
        """
        Quantities are able to directly convert to other units that
        have the same physical type.
        """
        if not self._include_easy_conversion_members:
            raise AttributeError(
                f"'{self.__class__.__name__}' object has no '{attr}' member")

        def get_virtual_unit_attribute():
            registry = get_current_unit_registry().registry
            to_unit = registry.get(attr, None)
            if to_unit is None:
                return None

            try:
                return self.unit.to(
                    to_unit, self.value, equivalencies=self.equivalencies)
            except UnitsError:
                return None

        value = get_virtual_unit_attribute()

        if value is None:
            raise AttributeError(
                f"{self.__class__.__name__} instance has no attribute '{attr}'")
        else:
            return value

    # =====================================================

    def __copy__(self) -> BaseQuantity[_ValueType]:
        q = self.__class__(copy.copy(self._magnitude), self._unit)
        return q

    def __deepcopy__(self, memo: dict[int, Any] | None) -> BaseQuantity[_ValueType]:
        q = self.__class__(
            copy.deepcopy(self._magnitude, memo),
            copy.deepcopy(self._unit, memo)
        )
        return q

    def __hash__(self) -> int:
        return hash(self.value) ^ hash(self.unit)

    # =====================================================
    # Operators
    # TODO!

    # Give warning for other >> self, since probably other << self was meant.
    def __rrshift__(self, other: Any):
        warnings.warn(">> is not implemented. Did you mean to convert "
                      "something to this quantity as a unit using '<<'?",
                      AstropyWarning)
        return NotImplemented

    def __pow__(self, other):
        return self.__class__(self._magnitude ** other, self._unit ** other)

    # =====================================================

    def __bool__(self):
        """Quantities should always be treated as non-False; there is too much
        potential for ambiguity otherwise.
        """
        warnings.warn('The truth value of a Quantity is ambiguous. '
                      'In the future this will raise a ValueError.',
                      AstropyDeprecationWarning)
        return True

    def __len__(self):
        return len(self._magnitude)

    # =====================================================
    # Numerical types

    def __float__(self):
        try:
            return float(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be '
                            'converted to Python scalars')

    def __int__(self):
        try:
            return int(self.to_value(dimensionless_unscaled))
        except (UnitsError, TypeError):
            raise TypeError('only dimensionless scalar quantities can be '
                            'converted to Python scalars')
