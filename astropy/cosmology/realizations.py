# Licensed under a 3-clause BSD style license - see LICENSE.rst

import sys
import warnings

from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from astropy.utils.state import ScienceState

from . import parameters
from .core import Cosmology, _get_subclasses

__all__ = ["default_cosmology"] + parameters.available

__doctest_requires__ = {"*": ["scipy"]}

# Pre-defined cosmologies. This loops over the parameter sets in the
# parameters module and creates a cosmology instance with the same name as the
# parameter set in the current module's namespace.
_subclasses = dict(_get_subclasses(Cosmology))
for key in parameters.available:
    params = getattr(parameters, key)

    # TODO! this will need refactoring again when: parameter I/O is JSON/ECSSV
    # and when metadata is allowed.
    if params["cosmology"] in _subclasses.keys():
        cosmo_cls = _subclasses[params["cosmology"]]

        par = dict()
        for k, v in params.items():
            if k not in cosmo_cls._init_signature.parameters:
                continue
            elif k == "H0":
                par["H0"] = u.Quantity(v, u.km / u.s / u.Mpc)
            elif k == "m_nu":
                par["m_nu"] = u.Quantity(v, u.eV)
            else:
                par[k] = v

        ba = cosmo_cls._init_signature.bind_partial(**par)
        cosmo = cosmo_cls(*ba.args, **ba.kwargs)

        docstr = "{} instance of {} cosmology\n\n(from {})"
        cosmo.__doc__ = docstr.format(key, cosmo_cls, params["reference"])

        setattr(sys.modules[__name__], key, cosmo)

        del cosmo_cls, par, k, v, ba, cosmo, docstr

# don't leave these variables floating around in the namespace
del key, params

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

    _value = "Planck18"

    @staticmethod
    def get_cosmology_from_string(arg):
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
    def validate(cls, value):
        if value is None:
            value = "Planck18"
        if isinstance(value, str):
            if value == "Planck18_arXiv_v2":
                warnings.warn(
                    f"{value} is deprecated in astropy 4.2, use Planck18 instead",
                    AstropyDeprecationWarning,
                )
            return cls.get_cosmology_from_string(value)
        elif isinstance(value, Cosmology):
            return value
        else:
            raise TypeError(
                "default_cosmology must be a string or Cosmology instance."
            )
