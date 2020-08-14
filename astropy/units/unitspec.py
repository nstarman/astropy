# -*- coding: utf-8 -*-

"""**DOCSTRING**.

Description.

"""

# __all__ = [
#     # classes
#     "",
#     # functions
#     "",
#     # other
#     "",
# ]


##############################################################################
# IMPORTS

# BUILT-IN

from abc import ABC
from types import MappingProxyType

# THIRD PARTY

import typing_extensions as T

# PROJECT-SPECIFIC

from .core import UnitBase, Unit
from .physical import get_physical_type, _physical_unit_mapping
from .quantity import Quantity
from .typing import UnitableType


##############################################################################
# Helper Functions


def isAnnotated(t):
    """is this thing an annotated type?"""
    if t == T.Annotated[t.__origin__, t.__metadata__]:
        return True
    return False


# /def

##############################################################################
# UnitSpecs
##############################################################################


class UnitSpecBase(ABC):
    """Base class for UnitSpecs."""

    def __init__(
        self,
        unit_or_physical_type: UnitableType,
        statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        super().__init__()
        self.unit_or_physical_type = unit_or_physical_type
        self.statictype = statictype

    # /def


# /class


# -------------------------------------------------------------------


class NullUnitSpec(UnitSpecBase):
    """Null UnitSpec. Returns value on call."""

    def __call__(value, *args, **kwargs):
        return value

    # /def


# /class


# -------------------------------------------------------------------

# registry of UnitSpec action options
_uspec_registry = {}


class UnitSpec(UnitSpecBase):
    """UnitSpec.

    Parameters
    ----------
    action : str
        works by mapping everything to appropriate UnitSpec subclass
        `__call__` method.

    """

    def __init_subclass__(cls, registry_name=None, **kwargs):
        super().__init_subclass__(**kwargs)

        # register in subclasses
        if registry_name is None:
            registry_name = cls.__name__.split("UnitSpec")[1].lower()

        if registry_name in _uspec_registry:
            raise Exception(
                "either rename class or pass unique `registry_name`"
            )  # TODO what type?

        _uspec_registry[registry_name] = cls

    # /def

    def __init__(
        self,
        physical_type: UnitableType,
        statictype: Quantity = Quantity,
        action: str = "validate",
    ):
        super().__init__(
            unit_or_physical_type=get_physical_type(physical_type),
            statictype=statictype,
        )

        if action not in _uspec_registry:
            raise KeyError(f"action must be one of {_uspec_registry.keys()}")
        self.action = action

        self._uspec_registry = MappingProxyType(_uspec_registry)

    # /def

    def __call__(self, value, **kwargs):
        return self._uspec_registry[self.action](value, **kwargs)

    # /def


# /class


# -------------------------------------------------------------------


class UnitSpecValidate(UnitSpecBase):
    """UnitSpec."""

    def __init__(
        self, physical_type: UnitableType, statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        super().__init__(
            unit_or_physical_type=get_physical_type(physical_type),
            statictype=statictype,
        )

    # /def

    def __call__(self, value):
        if not isinstance(value, self.statictype):  # includes subclasses
            raise TypeError

        if not get_physical_type(value) == self.unit_or_physical_type:
            raise ValueError

        return value


# /class


# -------------------------------------------------------------------


class UnitSpecConvert(UnitSpecBase):
    """UnitSpec."""

    def __init__(
        self,
        unit_or_physical_type: UnitableType,
        statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        physical_type = get_physical_type(unit_or_physical_type)
        if (
            unit_or_physical_type != physical_type
        ):  # keep only physcal type as str
            unit_or_physical_type = Unit(unit_or_physical_type)

        super().__init__(
            unit_or_physical_type=unit_or_physical_type, statictype=statictype,
        )

    # /def

    def __call__(self, value, statictype=None):

        if not hasattr(value, "unit"):
            raise TypeError

        if statictype is None:
            statictype = self.statictype

        if isinstance(self.unit_or_physical_type, str):
            physical_type_id = _physical_unit_mapping[
                self.unit_or_physical_type
            ]
            unit = Unit._from_physical_type_id(physical_type_id)
        else:
            unit = self.unit_or_physical_type

        return self.statictype(value, unit=unit, copy=False)


# /class


# -------------------------------------------------------------------


class UnitSpecValue(UnitSpecConvert, registry_name="to_value"):
    """UnitSpec."""

    def __call__(self, value, statictype=None):
        return super().__call__(value, statictype=statictype).to_value()


# /class


# -------------------------------------------------------------------


class UnitSpecAssign(UnitSpecBase, registry_name="from_value"):
    """UnitSpec."""

    def __call__(self, value, statictype=None):

        if statictype is None:
            statictype = self.statictype

        if isinstance(self.unit_or_physical_type, str):
            physical_type_id = _physical_unit_mapping[
                self.unit_or_physical_type
            ]
            unit = Unit._from_physical_type_id(physical_type_id)
        else:
            unit = self.unit_or_physical_type

        return self.statictype(value, unit=unit, copy=False)


# /class


##############################################################################
# END
