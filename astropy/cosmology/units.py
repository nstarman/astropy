# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Cosmological units and equivalencies.
"""  # (newline needed for unit summary)

import sys
import weakref
from types import MappingProxyType

import astropy.units as u
from astropy.units.utils import generate_unit_summary as _generate_unit_summary
from astropy.utils.compat import override__dir__

__all__ = ["littleh", "redshift",
           # equivalencies
           'CosmologyUnitEquivalencies',  # descriptor
           "dimensionless_redshift", "with_redshift", "with_H0"]

_ns = globals()


###############################################################################
# Cosmological Units

# This is not formally a unit, but is used in that way in many contexts, and
# an appropriate equivalency is only possible if it's treated as a unit.
redshift = u.def_unit(['redshift'], prefixes=False, namespace=_ns,
                      doc="Cosmological redshift.", format={'latex': r''})

# This is not formally a unit, but is used in that way in many contexts, and
# an appropriate equivalency is only possible if it's treated as a unit (see
# https://arxiv.org/pdf/1308.4150.pdf for more)
# Also note that h or h100 or h_100 would be a better name, but they either
# conflict or have numbers in them, which is disallowed
littleh = u.def_unit(['littleh'], namespace=_ns, prefixes=False,
                     doc='Reduced/"dimensionless" Hubble constant',
                     format={'latex': r'h_{100}'})


###############################################################################
# Equivalencies

class CosmologyUnitEquivalencies:
    """Cosmology-specific equivalencies.

    Parameters
    ----------
    redshift_temperature : bool (optional, keyword-only)
    with_H0 : bool (optional, keyword-only)

    atzkw : dict or None (optional, keyword-only)
        Keyword arguments for :func:`~astropy.cosmology.z_at_value`
    """

    __isabstractmethod__ = False  # needed to override __getattr__
    __slots__ = ["_equivs", "_equivs_attrs", "_atzkw",
                 "_parent_attr", "_parent_cls", "_parent_ref"]

    def __init__(self, *equivs, atzkw=None, **attrs):
        # available equivalencies and attribute mapping
        self._equivs = {f.__name__: f for f in equivs}  # applicable equivs
        self._equivs_attrs = attrs

        # kwargs for ``z_at_value``
        self._atzkw = atzkw

        # references to parent class and instance
        self._parent_attr = None  # set in __set_name__
        self._parent_cls = None
        self._parent_ref = None

    @property
    def available(self):
        """Available equivalencies."""
        return MappingProxyType(self._equivs)

    @override__dir__
    def __dir__(self):
        """Mix equivalencies into ``dir``."""
        return list(self._equivs.keys())

    def __str__(self):
        """Pretty string form."""
        namelead = (str(self._parent_cls) if self._parent_ref is None  # from class
                    else ((name if (name := self._parent.name) is not None  # instance w/ name
                           else "Cosmology instance")  # instance w/out name
                          + f" ({self._parent_cls.__qualname__!r})"))
        s = namelead + " unit equivalencies:\n"
        s += "\tavailable : " + ", ".join(f"'{k}'" for k in self.available.keys())
        if self._equivs_attrs:  # attr dict not empty
            s += "\n\tattrs : " + str(self._equivs_attrs)
        return s

    # ----------------------------
    # Serialization
    # needed b/c weakref cannot be pickled

    def __getstate__(self):
        state = {k: getattr(self, k) for k in self.__slots__}
        state["_parent_ref"] = None  # remove unpicklable weakref
        return state

    def __setstate__(self, state):
        for k, v in state.items():  # set like __dict__
            setattr(self, k, v)

    # ===============================================================
    # Descriptor stuff

    def __set_name__(self, objcls, name):
        self._parent_attr = name

    def __get__(self, obj, objcls):
        # accessed from a class
        if obj is None:
            self._parent_cls = objcls
            return self

        # accessed from an obj
        equivs = obj.__dict__.get(self._parent_attr)  # get from obj
        if equivs is None:  # hasn't been created on the obj
            descriptor = self.__class__(*self._equivs.values(),
                                        atzkw=self._atzkw, **self._equivs_attrs)
            descriptor._parent_cls = obj.__class__
            equivs = obj.__dict__[self._parent_attr] = descriptor

        # We set `_parent_ref` on every call, since if one makes copies of objs,
        # 'equivs' will be copied as well, which will lose the reference.
        equivs._parent_ref = weakref.ref(obj)
        return equivs

    @property
    def _parent(self):
        """Parent instance Cosmology."""
        return self._parent_ref() if self._parent_ref is not None else self._parent_cls

    # ===============================================================
    # Equivalencies

    def __getattr__(self, attr):
        """Get equivalency."""
        # not an attribute or 
        if attr not in self._equivs or self._parent_ref is None:
            raise AttributeError(f"equivalency {attr!r} does not apply to "
                                 f"cosmology {self._parent!r}")

        # get equivalency
        equiv_func = getattr(sys.modules[__name__], attr)
        equiv = equiv_func(getattr(self._parent, self._equivs_attrs[attr])
                           if attr in self._equivs_attrs else self._parent)
        return equiv

    @property
    def redshift(self):
        """Redshift equivalency :func:`~astropy.cosmology.units.with_redshift`."""
        if self._parent_ref is None:
            raise AttributeError("equivalency `with_redshift` does not apply "
                                 f"to cosmology class {self._parent!r}")

        return with_redshift(self._parent,
                             Tcmb="redshift_temperature" in self.available,
                             atzkw=self._atzkw)


###############################################################################
# Equivalencies


def dimensionless_redshift():
    """Allow redshift to be 1-to-1 equivalent to dimensionless.

    It is special compared to other equivalency pairs in that it
    allows this independent of the power to which the redshift is raised,
    and independent of whether it is part of a more complicated unit.
    It is similar to u.dimensionless_angles() in this respect.
    """
    return u.Equivalency([(redshift, None)], "dimensionless_redshift")


def redshift_temperature(cosmology=None, **atzkw):
    """Convert quantities between redshift and CMB temperature.

    Care should be taken to not misinterpret a relativistic, gravitational, etc
    redshift as a cosmological one.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology`, str, or None, optional
        A cosmology realization or built-in cosmology's name (e.g. 'Planck18').
        If None, will use the default cosmology
        (controlled by :class:`~astropy.cosmology.default_cosmology`).
    **atzkw
        keyword arguments for :func:`~astropy.cosmology.z_at_value`

    Returns
    -------
    `~astropy.units.equivalencies.Equivalency`
        Equivalency between redshift and temperature.
    """
    from astropy.cosmology import default_cosmology, z_at_value

    # get cosmology: None -> default and process str / class
    cosmology = cosmology if cosmology is not None else default_cosmology.get()
    with default_cosmology.set(cosmology):  # if already cosmo, passes through
        cosmology = default_cosmology.get()

    def z_to_Tcmb(z):
        return cosmology.Tcmb(z)

    def Tcmb_to_z(T):
        return z_at_value(cosmology.Tcmb, T << u.K, **atzkw)

    return u.Equivalency([(redshift, u.K, z_to_Tcmb, Tcmb_to_z)], "redshift_temperature",
                         {'cosmology': cosmology})


def with_redshift(cosmology=None, *, Tcmb=True, atzkw=None):
    """Convert quantities between measures of cosmological distance.

    Note: by default all equivalencies are on and must be explicitly turned off.
    Care should be taken to not misinterpret a relativistic, gravitational, etc
    redshift as a cosmological one.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology`, str, or None, optional
        A cosmology realization or built-in cosmology's name (e.g. 'Planck18').
        If None, will use the default cosmology
        (controlled by :class:`~astropy.cosmology.default_cosmology`).
    Tcmb : bool (optional, keyword-only)
        Whether to create a CMB temperature <-> redshift equivalency, using
        ``Cosmology.Tcmb``. Default is False.
    atzkw : dict or None (optional, keyword-only)
        keyword arguments for :func:`~astropy.cosmology.z_at_value`

    Returns
    -------
    `~astropy.units.equivalencies.Equivalency`
        With equivalencies between redshift and temperature.
    """
    from astropy.cosmology import default_cosmology, z_at_value

    # get cosmology: None -> default and process str / class
    cosmology = cosmology if cosmology is not None else default_cosmology.get()
    with default_cosmology.set(cosmology):  # if already cosmo, passes through
        cosmology = default_cosmology.get()

    atzkw = atzkw if atzkw is not None else {}
    equivs = []  # will append as built

    # -----------
    # CMB Temperature <-> Redshift

    if Tcmb:
        equivs.extend(redshift_temperature(cosmology, **atzkw))

    # -----------

    return u.Equivalency(equivs, "with_redshift",
                         {'cosmology': cosmology, 'Tcmb': Tcmb})


# ===================================================================

def with_H0(H0=None):
    """
    Convert between quantities with little-h and the equivalent physical units.

    Parameters
    ----------
    H0 : `~astropy.units.Quantity` ['frequency'], |Cosmology|, or None, optional
        The value of the Hubble constant to assume. If a |Quantity|, will
        assume the quantity *is* ``H0``. If `None` (default), use the ``H0`
        attribute from :mod:`~astropy.cosmology.default_cosmology`.

    References
    ----------
    For an illuminating discussion on why you may or may not want to use
    little-h at all, see https://arxiv.org/pdf/1308.4150.pdf
    """
    if H0 is None:
        from .realizations import default_cosmology
        H0 = default_cosmology.get().H0
    elif not isinstance(H0, u.Quantity):  # a Cosmology
        H0 = H0.H0

    h100_val_unit = u.Unit(100/(H0.to_value((u.km/u.s)/u.Mpc)) * littleh)

    return u.Equivalency([(h100_val_unit, None)], "with_H0", kwargs={"H0": H0})


# ===================================================================
# Enable the set of default equivalencies.
# If the cosmology package is imported, this is added to the list astropy-wide.

u.add_enabled_equivalencies(dimensionless_redshift())


# =============================================================================
# DOCSTRING

# This generates a docstring for this module that describes all of the
# standard units defined here.
if __doc__ is not None:
    __doc__ += _generate_unit_summary(_ns)
