# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import warnings
from types import MappingProxyType

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points

from astropy import units as u
from astropy.utils.decorators import deprecated
from astropy.utils.exceptions import AstropyDeprecationWarning, AstropyUserWarning
from astropy.utils.state import ScienceState

from . import parameters
from .core import _COSMOLOGY_CLASSES, Cosmology

__all__ = ["default_cosmology"] + parameters.available

__doctest_requires__ = {"*": ["scipy"]}


# Pre-defined cosmologies. This loops over the parameter sets in the
# parameters module and creates a corresponding cosmology instance
for key in parameters.available:
    params = getattr(parameters, key)  # get parameters dictionary
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
    The default cosmology realization to use.  To change it::

        >>> from astropy.cosmology import default_cosmology, WMAP7
        >>> with default_cosmology.set(WMAP7):
        ...     # WMAP7 cosmology in effect
        ...     pass

    Or, you may use a string::

        >>> with default_cosmology.set('WMAP7'):
        ...     # WMAP7 cosmology in effect
        ...     pass

    To get the default Cosmology, or a specific cosmology::

        >>> default_cosmology.get()
        >>> default_cosmology.get('WMAP9')

    """

    _latest_value = "Planck18"  # marks the most up-to-date cosmology
    _value = None  # the current value

    # Registry of realizations. Starts with the built-ins.
    _registry = {k: getattr(sys.modules[__name__], k)
                 for k in parameters.available}
    available = _registry.keys()  # names of registered

    @classmethod
    def register(cls, cosmology, key=None):
        """Register a Cosmology instance for use in ``default_cosmology``.

        Parameters
        ----------
        cosmology : `~astropy.cosmology.Cosmology` subclass instance
        key : str or None, optional
            The registry key for ``cosmology``. If None (default), the key is
            ``cosmology.name``.

        Raises
        ------
        ValueError
            If the name is "no_default" or "latest" or is for a built-in
            cosmology realization (see ``cosmology.parameters.available``).
        """
        # check it's a Cosmology
        if not isinstance(cosmology, Cosmology):
            raise TypeError(
                f"{cosmology} must be an instance of a Cosmology subclass")

        # check if valid registry name. Invalid if 'latest', 'no_default', or
        # any of the built-in realizations (e.g. Planck18). Warn if overwriting
        # a non-builtin.
        name = cosmology.name if key is None else key
        if name in ("latest", "no_default"):
            raise ValueError("Name cannot be one of {'latest', 'no_default'}.")
        elif name in parameters.available:  # an Astropy built-in
            raise ValueError(f"Cannot override built-in realization {name}.")
        elif name in cls._registry:  # registered, but not built-in
            warnings.warn(f"Registering cosmology realization '{name}', "
                          "overwriting existing registered realization.",
                          category=AstropyUserWarning)
        # register
        cls._registry[name] = cosmology

    @deprecated("v5.0", alternative="`get`")
    @classmethod
    def get_cosmology_from_string(cls, arg):
        """ Return a cosmology instance from a string.
        """
        if arg == "no_default":
            cosmo = None
        else:
            try:
                cosmo = getattr(sys.modules[__name__], arg)
            except AttributeError:
                s = "Unknown cosmology '{}'. Valid cosmologies:\n{}".format(
                    arg, parameters.available
                )
                raise ValueError(s)
        return cosmo

    @classmethod
    def get(cls, name=None):
        """Return a cosmology instance from a string.

        Parameters
        ----------
        name : str or None, optional
            The registry key (str) for ``cosmology``.
            If None (default), the current state is returned.

        Returns
        -------
        :class:`~astropy.cosmology.Cosmology` or None
            None if 'name' is "no_default",
            most up-to-date cosmology realization if 'name' is "latest".
            Cosmology instance if 'name' is key in registry.

        Raises
        ------
        KeyError
            If name is not None, 'no_default', 'latest', or key in registry.
        """
        # If no name, get the current science state value.
        # this uses ``validate(cls._value)``, which will redirect here,
        # but with the correct ``name`` set. No infinite loops.
        if name is None:
            return super().get()
        # option for no default
        elif name == "no_default":
            return None

        # Resolve the meaning of 'latest': 
        if name == 'latest':
            name = cls._latest_value
        # Get the state from the registry.
        try:
            cosmo = cls._registry[name]
        except KeyError:
            s = (f"Unknown cosmology '{name}'. Valid cosmologies:\n"
                 f"{cls._registry.keys()}")
            raise ValueError(s) from None

        return cosmo

    @classmethod
    def validate(cls, value):
        """Validate the value and convert it to its native type, if necessary.

        Returns
        -------
        :class:`~astropy.cosmology.Cosmology` or None
            None if 'name' is "no_default",
            most up-to-date cosmology realization if 'name' is "latest".
            Cosmology instance if 'name' is key in registry.
        """
        if value is None:
            value = cls._latest_value

        if isinstance(value, str):
            if value == "Planck18_arXiv_v2":
                warnings.warn(
                    f"{value} is deprecated since Astropy 4.2 and will be "
                    "removed in 5.1. Use 'Planck18' instead.",
                    AstropyDeprecationWarning,
                )
            cosmo = cls.get(name=value)
        elif isinstance(value, Cosmology):
            cosmo = value
        else:
            raise TypeError("default_cosmology must be a string or Cosmology "
                            f"instance, not {type(value)}.")
        return cosmo
