# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
# STDLIB
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar, runtime_checkable

# THIRD PARTY
import numpy as np

from astropy.units.core import Unit

if TYPE_CHECKING:
    from astropy.units.core import UnitBase  # noqa: F401
    from astropy.units.equivalencies import Equivalency  # noqa: F401
    from astropy.units.quantity.info import QuantityInfo  # noqa: F401

    # from typing_extensions import Self  # noqa: F401
    Self = Any


__all__ = ["DuckArray", "DuckQuantity", "QuantityBase"]


_ValueType = TypeVar("_ValueType", covariant=True)


@runtime_checkable
class DuckArray(Protocol):

    # def __duckarray__(self: Self) -> Self: ...  # TODO! whn NEP 30

    # def __array_function__(self)

    @property
    def dtype(self) -> np.dtype: ...

    @property
    def shape(self) -> tuple[int, ...]: ...

    @property
    def ndim(self) -> int: ...


@runtime_checkable
class DuckQuantity(Protocol[_ValueType]):

    @property
    def value(self) -> _ValueType: ...

    @property
    def unit(self) -> UnitBase: ...

    @property
    def equivalency(self) -> list[Equivalency]: ...

    @property
    def info(self) -> QuantityInfo: ...

    def to(self: Self, unit: UnitBase | str, equivalencies: list[Equivalency], copy=True) -> Self: ...

    def to_value(self, unit: UnitBase | str, equivalencies: list[Equivalency]) -> _ValueType: ...


class QuantityBase(Generic[_ValueType], metaclass=ABCMeta):
    pass
