# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

import abc
from numbers import Number
from collections.abc import ItemsView
from types import MappingProxyType

import numpy as np

from astropy.utils.decorators import classproperty

from .core import Unit, UnitBase, UnitConversionError, UnitsError
from .physical import _physical_unit_mapping, _unit_physical_mapping, get_physical_type
from .quantity import Quantity

__all__ = ["UnitSpec", "CompoundUnitSpec", "NullUnitSpec"]

# registry of UnitSpec action options
_ACTION_USPEC_REGISTRY = {}
_USPEC_ACTION_REGISTRY = {}


class UnitSpecBase(metaclass=abc.ABCMeta):
    """Abstract base class for Unit Specification.

    Use this in ``isinstance`` checks.

    Parameters
    ----------
    unit : `astropy.units.UnitBase` instance
    qcls : type

    """

    def __init__(self, target, qcls=Quantity, strict=False, **kw):
        self.target = target
        self.qcls = qcls
        self.strict = strict

    @classmethod
    def get_action(cls, kls=None):
        """Get action associated with a unit specification class."""
        return _USPEC_ACTION_REGISTRY[kls or cls]

    action = classproperty(get_action, lazy=True)

    @abc.abstractmethod
    def __call__(self, value, **kwargs):
        raise NotImplementedError

    # =====================

    def __add__(self, other):
        # check if other is UnitSpecBase
        if not isinstance(other, UnitSpecBase):
            raise TypeError(f"{other!r} must be {UnitSpecBase!r}")

        return CompoundUnitSpec(self, other)

    def __repr__(self):
        return f"{self.__class__.__qualname__}({self.target}, qcls={self.qcls.__name__})"


class CompoundUnitSpec(UnitSpecBase):

    def __init__(self, *specs):
        # TODO! flatten specs to allow for CompoundUnitSpec(CompoundUnitSpec())
        # also check if there are conflicting specs
        self._specs = specs

    def __call__(self, value, **kwargs):
        # TODO! better iteration over specs
        for spec in self._specs:
            try:
                out = spec(value, **kwargs)
            except (UnitConversionError, TypeError):
                pass
            else:
                return out

        raise TypeError("All unitspecs failed")  # TODO! better error message

    # =====================

    def __iadd__(self, other):
        # check if other is UnitSpecBase
        if not isinstance(other, UnitSpecBase):
            raise TypeError(f"{other!r} must be {UnitSpecBase!r}")

        # allow for other to be CompoundUnitSpec
        if isinstance(other, CompoundUnitSpec):
            other = other._specs
        else:
            other = (other, )

        self._specs = self._specs + other
        return self

    def __repr__(self):
        # TODO! nicely split long lines
        return f"{self.__class__.__qualname__}{self._specs!r}"

# =============================================================================


class UnitSpec(UnitSpecBase):
    """Unit Specification.

    Parameters
    ----------
    target
    action : str
        works by mapping everything to appropriate UnitSpec subclass
        `__call__` method.
    qcls

    """

    def __init_subclass__(cls, action, **kwargs):
        # check registry
        if action in _ACTION_USPEC_REGISTRY:
            raise KeyError(f"`{action}` already in registry.")

        # register subclass and action
        _ACTION_USPEC_REGISTRY[action] = cls
        _USPEC_ACTION_REGISTRY[cls] = action

        # set the "action" property
        # overrides the properties below, which gets the action from the uspec
        cls.action = classproperty(fget=cls.get_action, lazy=True,
                                   doc=f"{action.capitalize()} action.")

    def __new__(cls, target=None, action="validate", qcls=Quantity, **kw):
        # First check if target is a UnitSpec. This class doesn't convert, so
        # those pass thru directly. Note: UnitSpec subclasses can do whatever.
        if isinstance(target, UnitSpecBase):
            return target

        # second check it's a valid action...
        elif action not in _ACTION_USPEC_REGISTRY:
            raise KeyError(f"action {action!r} must be one of {_ACTION_USPEC_REGISTRY.keys()}")

        # if so, create the UnitSpec for that action.
        self = super().__new__(cls)
        if cls is UnitSpec:
            # UnitSpec wraps its subclasses so only UnitSpec need ever be imported.
            # Need to determine which subclass to wrap
            self._uspec = _ACTION_USPEC_REGISTRY[action](target, qcls=qcls)

        return self

    def __call__(self, value, **kwargs):
        # pass through to wrapped UnitSpec
        return self._uspec(value, **kwargs)

    @property
    def action(self):
        """Unit Specification."""
        return self._uspec.action

    @action.setter
    def action(self, value):
        if value != self._uspec.action:
            self._uspec = self._ACTION_USPEC_REGISTRY[value](self.target, qcls=self.qcls)

    # =====================

    def __repr__(self):
        if self.__class__ is UnitSpec:
            return repr(self._uspec)
        return f"UnitSpec({self.target!r}, qcls={self.qcls.__qualname__}, action={self.action!r})"


# -----------------------------------------------------------------------------

class NullUnitSpec(UnitSpecBase):
    """Null Unit Specification. Returns value on call."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, value, *args, **kwargs):
        """Returns 'value' unchanged."""
        return value

    def __repr__(self):
        return f"{self.__class__.__qualname__}()"


# -----------------------------------------------------------------------------

class UnitSpecValidatePhysicalType(UnitSpec, action="validate"):
    """UnitSpec to validate physical type.
    
    Parameters
    ----------
    physical_type : `~astropy.units.PhysicalType`-like
        e.g. the string 'length'
    qcls : type
        The :class:`~astropy.units.Quantity` class.
        Can be used to check that an input is not only the correct physical
        type, but also the correct Quantity type.

    strict : bool
        Whether to be strict about stuff like numerics (`~number.Number` or
        `~numpy.ndarray`) being equivalent to dimensionless quantities.

    """

    def __init__(self, physical_type, qcls=Quantity, strict=False, **kw):
        # make sure it's actually a physical type. this will catch any errors.
        ptype = get_physical_type(physical_type)
        super().__init__(ptype, qcls=qcls, strict=strict, **kw)

    def __call__(self, value, strict=None, **kw):
        # what if 'value' was just a number? if not 'strict', numbers will be
        # treated as dimensionless quantities.
        strict = self.strict if strict is None else strict
        if (not strict
            and (self.target == "dimensionless")
            and (isinstance(value, Number)
                 or (isinstance(value, np.ndarray)
                     and np.issubdtype(value.dtype, np.number)))):
            pass  # He is the `~astropy.units.one`.

        # checks that 'value' is the the correct class type
        elif not isinstance(value, self.qcls):
            raise TypeError(f"{value!r} is not qcls {self.qcls!r} (or subclass).")
        elif strict and value.__class__ is not self.qcls:
            raise TypeError(f"{value!r} is not type {self.qcls!r}.")

        # check units is the correct physical type, including any enabled
        # equivalencies. TODO! a cleaner check
        elif not value.unit.is_equivalent(self.target._unit):
            raise UnitConversionError(f"`{value}` is not equivalent to type '{self.target}'")
        else:
            pass  # passes all checks!

        return value

    # =====================

    def __repr__(self):
        return (f"UnitSpec('{self.target}', qcls={self.qcls.__qualname__}, "
                f"action={self.action!r}, strict={self.strict})")


# -----------------------------------------------------------------------------

class UnitSpecConvertToUnits(UnitSpec, action="convert"):
    """Convert input to target units.

    Parameters
    ----------
    unit : unit-like
        Not Physical Type
    qcls : type
    strict : bool
    """

    def __init__(self, unit, qcls=Quantity, strict=True, **kw):
        unit = Unit(unit)  # make sure it's a Unit. catches any errors.
        super().__init__(unit, qcls=qcls, strict=strict, **kw)

    def __call__(self, value, strict=None, **kw):
        """Convert input to target units.

        Parameters
        ----------
        value
        strict : bool

        Returns
        -------
        `~astropy.units.Quantity`

        Raises
        ------
        TypeError
            If 'strict' and 'value' is not `~astropy.units.Quantity`
        """
        # Determine strictness. if strict, only permit Quantity, not number.
        strict = self.strict if strict is None else strict
        if strict and not isinstance(value, self.qcls):
            raise TypeError(f"{value!r} is not type {self.qcls!r}")
        # convert value to desired quantity
        return self.qcls(value, unit=self.target, copy=False)


# -------------------------------------------------------------------


class UnitSpecConvertToValue(UnitSpecConvertToUnits, action="to_value"):
    """Convert input to value in target units.

    Parameters
    ----------
    unit : unit-like
        Not Physical Type
    qcls : type
    strict : bool
    """

    def __call__(self, value, strict=None, **kw):
        """Convert input to value in target units.
    
        Parameters
        ----------
        value
        strict : bool
    
        Returns
        -------
        `~astropy.units.Quantity`
    
        Raises
        ------
        TypeError
            If 'strict' and 'value' is not `~astropy.units.Quantity`
        """
        return super().__call__(value, strict=strict, **kw).to_value()


# -------------------------------------------------------------------


class UnitSpecAssignUnits(UnitSpecConvertToUnits, action="from_value"):
    """Assign input target units.
    
    Equivalent to UnitSpec 'convert', with strictness set to False.

    Parameters
    ----------
    unit : unit-like
        Not Physical Type
    qcls : type
    """

    def __init__(self, unit, qcls=Quantity, **kw):
        kw.pop("strict", None)  # make sure not present
        super().__init__(unit, qcls=qcls, strict=False, **kw)

    def __call__(self, value, **kw):
        """Assign input target units.
    
        Parameters
        ----------
        value
    
        Returns
        -------
        `~astropy.units.Quantity`
        """
        kw.pop("strict", None)  # make sure not present
        super().__call__(value, strict=False, **kw)
