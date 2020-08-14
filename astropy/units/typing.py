# -*- coding: utf-8 -*-

"""typing for :mod:`~astropy.units`."""

__all__ = [
    "UnitableType",
]


##############################################################################
# IMPORTS

# THIRD PARTY

import typing_extensions as T

# PROJECT-SPECIFIC

from .core import UnitBase
from .quantity import Quantity


##############################################################################
# TYPES
##############################################################################


UnitableType = T.TypeVar("UnitableType", UnitBase, Quantity, str)
"""Types that work with Astropy [astropy]_ :class:`~astropy.units.Unit`.

Subclasses of Unit, Quantity.

References
----------
.. [astropy] Astropy Collaboration et al., 2018, AJ, 156, 123.

Examples
--------

    >>> x: UnitableType = 10 * u.km
    >>> Unit(x)
    10 km

"""


# -------------------------------------------------------------------


##############################################################################
# END
