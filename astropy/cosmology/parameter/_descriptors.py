# Licensed under a 3-clause BSD style license - see LICENSE.rst

from __future__ import annotations

__all__ = []

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

from astropy.utils.compat.misc import PYTHON_LT_3_10

if TYPE_CHECKING:
    from astropy.cosmology.core import Cosmology


_dataclass_kwargs = {} if PYTHON_LT_3_10 else {"slots": True}


@dataclass(frozen=True, **_dataclass_kwargs)
class ParameterDescriptor:
    """Immutable mapping of the Parameters.

    If accessed from the class, this returns a mapping of the Parameter
    objects themselves.  If accessed from an instance, this returns a
    mapping of the values of the Parameters.
    """

    def __get__(
        self, instance: Cosmology | None, owner: type[Cosmology] | None
    ) -> MappingProxyType[str, Any]:
        # called from the class
        if instance is None:
            return owner._parameters_
        # called from the instance
        return MappingProxyType(
            {n: getattr(instance, n) for n in instance._parameters_}
        )

    def __set__(self, instance: Any, value: Any) -> NoReturn:
        raise AttributeError(f"cannot set 'parameters' of {instance!r}.")


@dataclass(frozen=True, **_dataclass_kwargs)
class DerivedParameterDescriptor:
    """Immutable mapping of the derived Parameters.

    If accessed from the class, this returns a mapping of the derived
    Parameter objects themselves.  If accessed from an instance, this
    returns a mapping of the values of the derived parameters.
    """

    def __get__(self, instance, owner):
        # called from the class
        if instance is None:
            return owner._parameters_derived_
        # called from the instance
        return MappingProxyType(
            {n: getattr(instance, n) for n in instance._parameters_derived_}
        )

    def __set__(self, instance, value):
        raise AttributeError(f"cannot set 'derived_parameters' of {instance!r}.")
