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

# THIRD PARTY

import typing_extensions as T

# PROJECT-SPECIFIC

from .core import UnitBase
from .quantity import Quantity


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
        unit_or_physical_type: T.Union[UnitBase, str, Quantity],
        dtype: Quantity = Quantity,
    ):
        """Initialize class."""
        super().__init__()
        self.unit_or_physical_type = unit_or_physical_type
        self.dtype = dtype

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


class UnitSpec(UnitSpecBase):
    """UnitSpec."""

    pass


# /class


# class UnitSpecValidate(UnitSpecBase):
#     """Base class for UnitSpecs."""

#     def __init__(
#         self,
#         unit_or_physical_type: T.Union[UnitBase, str, Quantity],
#         dtype: Quantity = Quantity,
#     ):
#         """Initialize class."""
#         super().__init__(unit_or_physical_type=unit_or_physical_type, dtype=dtype)


# # /class


# # -------------------------------------------------------------------


# class UnitSpecCast(UnitSpecBase):
#     """Base class for UnitSpecs."""

#     def __init__(
#         self,
#         unit_or_physical_type: T.Union[UnitBase, str, Quantity],
#         dtype: Quantity = Quantity,
#     ):
#         """Initialize class."""
#         super().__init__(unit_or_physical_type=unit_or_physical_type, dtype=dtype)


# # /class


##############################################################################
# END
