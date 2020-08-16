from abc import ABC
from types import MappingProxyType
import typing as T
from collections.abc import ItemsView

# THIRD PARTY

from typing_extensions import Annotated

# PROJECT-SPECIFIC

import astropy.units as u
from astropy.units.core import UnitBase, Unit
from astropy.units.physical import (
    get_physical_type,
    _physical_unit_mapping,
    _unit_physical_mapping,
)
from astropy.units.quantity import Quantity
from astropy.units.typing import UnitableType
from astropy.utils.decorators import classproperty


##############################################################################
# Helper Area

# TODO Move
def isAnnotated(t) -> bool:
    """is this thing an annotated type?"""
    if not hasattr(t, "__origin__") or not hasattr(t, "__metadata__"):
        return False

    if t == Annotated[t.__origin__, t.__metadata__]:
        return True

    return False


# registry of UnitSpec action options
_action_uspec_registry = {}
_uspec_action_registry = {}


##############################################################################
# UnitSpecs
##############################################################################


class UnitSpecBase(ABC):
    """Base class for UnitSpecs."""

    def __new__(cls, unit, *args, **kwargs):
        self = super().__new__(cls)
        self._orig_value = unit  # save value passed at initialization

        return self

    def __init__(
        self, unit: UnitableType, statictype: Quantity = Quantity, **kwargs
    ):
        """Initialize class."""
        super().__init__()

        self.statictype = statictype
        self.unit = unit

    @classmethod
    def get_action(cls, kls=None):
        """"""
        return ItemsView(_uspec_action_registry[kls or cls])

    @classproperty(lazy=True)
    def action(cls):
        """read-only of action description."""
        return cls.get_action()


# /class


# -------------------------------------------------------------------


class NullUnitSpec(UnitSpecBase):
    """Null UnitSpec. Returns value on call."""

    def __call__(value, *args, **kwargs):
        return value


# /class


# -------------------------------------------------------------------


class UnitSpec(UnitSpecBase):
    """UnitSpec.

    Parameters
    ----------
    action : str
        works by mapping everything to appropriate UnitSpec subclass
        `__call__` method.

    """

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, value):
        if value != self._action:
            self._uspec = self._action_uspec_registry[value](
                self.unit, statictype=self.statictype
            )
        self._action = value

    def __init_subclass__(
        cls, registry_name: T.Optional[str] = None, **kwargs
    ):
        #         cls.action = classproperty(fget=cls.get_action, doc="TEST2")
        super().__init_subclass__(**kwargs)

        # register in subclasses
        if registry_name is None:
            registry_name = cls.__name__.split("UnitSpec")[1].lower()

        if registry_name in _action_uspec_registry:
            # TODO what type of Exception?
            raise Exception(f"`{registry_name}` already in registry.")

        _action_uspec_registry[registry_name] = cls
        _uspec_action_registry[cls] = registry_name

        # set the action to be read-only
        cls.action = classproperty(fget=cls.get_action, doc="TEST2")

    def __new__(
        cls,
        unit: UnitableType,
        statictype: Quantity = Quantity,
        action: str = "validate",
    ):
        if action not in _action_uspec_registry:
            raise KeyError(
                f"Action {action} must be one of {_action_uspec_registry.keys()}"
            )

        self = super().__new__(cls, unit, statictype, action)
        if cls is UnitSpec:
            self._action = action
            self._uspec = _action_uspec_registry[action](
                unit, statictype=statictype
            )

        return self

    def __init__(
        self,
        unit: UnitableType,
        statictype: Quantity = Quantity,
        action: str = "validate",
    ):
        if unit not in _unit_physical_mapping.keys():
            unit = Unit(unit)

        super().__init__(unit=unit, statictype=statictype)

    def __call__(self, value, **kwargs):
        return self._uspec(value, **kwargs)


# /class


# -------------------------------------------------------------------


class UnitSpecValidate(UnitSpec):
    """UnitSpec."""

    def __init__(
        self, physical_type: UnitableType, statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        if physical_type in _unit_physical_mapping.keys():
            pass
        elif hasattr(physical_type, "unit"):  # quantity
            physical_type = get_physical_type(physical_type.unit)
        else:
            physical_type = get_physical_type(Unit(physical_type))

        super().__init__(
            unit=physical_type, statictype=statictype,
        )

    def __call__(self, value):
        if not isinstance(value, self.statictype):  # includes subclasses
            raise TypeError

        if not get_physical_type(value.unit) == self.unit:
            raise UnitConversionError(f"{value} is not type {self.unit}")

        return value


# /class


# -------------------------------------------------------------------


class UnitSpecConvert(UnitSpec):
    """UnitSpec."""

    def __init__(
        self, unit: UnitableType, statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        if unit in _unit_physical_mapping.keys():
            pass
        # keep only physical type as str
        else:
            unit = Unit(unit)

        super().__init__(
            unit=unit, statictype=statictype,
        )

    def __call__(self, value, statictype=None):

        if not hasattr(value, "unit"):
            raise TypeError

        if statictype is None:
            statictype = self.statictype

        if isinstance(self.unit, str):
            physical_type_id = _unit_physical_mapping[self.unit]
            unit = Unit._from_physical_type_id(physical_type_id)
        else:
            unit = self.unit

        return self.statictype(value, unit=unit, copy=False)


# -------------------------------------------------------------------


class UnitSpecValue(UnitSpecConvert, registry_name="to_value"):
    """UnitSpec."""

    def __init__(
        self, unit: UnitableType, statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        if unit in _unit_physical_mapping.keys():
            raise ValueError("Cannot be physical type")

        super().__init__(unit=unit, statictype=statictype)

    def __call__(self, value, statictype=None):
        return super().__call__(value, statictype=statictype).to_value()


# -------------------------------------------------------------------


class UnitSpecAssign(UnitSpec, registry_name="from_value"):
    """UnitSpec."""

    def __init__(
        self, unit: UnitableType, statictype: Quantity = Quantity,
    ):
        """Initialize class."""
        if unit in _unit_physical_mapping.keys():
            raise ValueError("Cannot be physical type")

        super().__init__(unit=unit, statictype=statictype)


    def __call__(self, value, statictype=None):

        if not hasattr(value, "unit"):
            value = value * self.unit

        # TODO allow no conversion if value has right unit type?

        return self.statictype(value, unit=self.unit, copy=False)
