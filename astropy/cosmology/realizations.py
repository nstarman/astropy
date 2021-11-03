# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import warnings

from astropy import units as u
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.utils.state import ScienceState

from . import parameters
from .core import _COSMOLOGY_CLASSES, Cosmology

__all__ = ["default_cosmology"] + list(parameters.available)

__doctest_requires__ = {"*": ["scipy"]}


# Pre-defined cosmologies. This loops over the parameter sets in the
# parameters module and creates a corresponding cosmology instance
for key in parameters.available:
    params = dict(getattr(parameters, key))  # get parameters dict (copy)
    params.setdefault("name", key)
    # make cosmology
    cosmo = Cosmology.from_format(params, format="mapping", move_to_meta=True)
    cosmo.__doc__ = (f"{key} instance of {cosmo.__class__.__qualname__} "
                     f"cosmology\n(from {cosmo.meta['reference']})")
    # put in this namespace
    setattr(sys.modules[__name__], key, cosmo)

del key, params, cosmo  # clean the namespace

#########################################################################
# The science state below contains the current cosmology.
#########################################################################


class default_cosmology(ScienceState):
    """
    The default cosmology to use.  To change it::

        >>> from astropy.cosmology import default_cosmology, WMAP7
        >>> with default_cosmology.set(WMAP7):
        ...     # WMAP7 cosmology in effect
        ...     pass

    Or, you may use a string::

        >>> with default_cosmology.set('WMAP7'):
        ...     # WMAP7 cosmology in effect
        ...     pass
    """

    _default_value = "Planck18"
    _value = "Planck18"

    @classmethod
    def get(cls, key=None):
        """Get the science state value of ``key``.

        If ``key`` is None get the current value.

        Parameters
        ----------
        key : str
            
        """
        if key is None:
            key = cls._value

        if isinstance(key, str):
            try:
                value = getattr(sys.modules[__name__], key)
            except AttributeError:
                raise ValueError(f"Unknown cosmology {key!r}. "
                                 f"Valid cosmologies:\n{parameters.available}")
        elif isinstance(key, Cosmology):
            value = key
        else:
            raise TypeError("'key' must be must be None, a string, or Cosmology instance, "
                            f"not {type(key)}.")

        return cls.validate(value)

    @deprecated("5.0", alternative="get")
    @classmethod
    def get_cosmology_from_string(cls, arg):
        """Return a cosmology instance from a string."""
        return cls.get(arg)

    @classmethod
    def validate(cls, value):
        if value is None:
            value = cls._default_value

        if isinstance(value, str):
            value = cls.get(value)
        elif not isinstance(value, Cosmology):
            raise TypeError("default_cosmology must be a string or Cosmology instance.")

        return value
