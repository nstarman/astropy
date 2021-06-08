# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import operator

import numpy as np

import astropy.units as u
from astropy.utils import ShapedLikeNDArray

from .realizations import default_cosmology

__all__ = ["CosmologicalRedshift"]

__doctest_requires__ = {"*": ["scipy"]}


################################################################################
# Redshift Object


class CosmologicalRedshift(u.SpecificTypeQuantity):
    """Cosmological Redshift.

    Parameters
    ----------
    z : array-like or quantity-like ['dimensionless', 'redshift']

    """

    _equivalent_unit = u.one
    _include_easy_conversion_members = True
    _default_unit = u.one

    def __new__(cls, value=None, scale_factor=None, dtype=None, copy=True,
                order=None, subok=False, ndmin=0, **kwargs):
        # TODO! switch to << redshift_unit, when PR merged
        # TODO! use equivalencies, when PR merged, to accept other units
        _ns = locals()
        inputs = {k for k in ("value", "scale_factor") if _ns[k] is not None}
        if len(inputs) != 1:
            msg = ("Should give one one of `value` or `scale_factor` in "
                   "CosmologicalRedshift constructor.")
            if len(inputs) > 1:
                msg += f"Values were given for {inputs}."
            raise ValueError(msg)
        elif value is not None:
            pass
        elif scale_factor is not None:
            value = np.power(scale_factor, -1.0) - 1.0

        return super().__new__(cls, value << u.one, unit=u.one, dtype=dtype,
                               copy=copy, order=order, subok=subok, ndmin=ndmin)

    @property
    def dimensionless(self):
        """The value of this instance with dimensionless units."""
        # this equivalency is enabled by default.
        # with u.set_enabled_equivalencies(u.dimensionless_redshift()):
        # return self._z.to(u.one)
        return self.to(u.one)

    # --------------------------------------------------
    # Equivalencies
    # TODO when PR merged

    # --------------------------------------------------
    # Distances

    @classmethod
    def _resolve_cosmology(cls, cosmology):
        if cosmology is None:
            cosmology = default_cosmology.get()
        return cosmology

    def _calculate_scale_factor(self, cosmology=None):
        """Calculate the scale factor at specified redshift.

        The scale factor is defined as :math:`a = 1 / (1 + z)`.
        """
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.scale_factor(self.value) << u.one

    scale_factor = property(_calculate_scale_factor)

    def _calculate_lookback_distance(self, cosmology=None):
        """The lookback distance in Mpc at each input redshift.

        The lookback distance is the light travel time distance to a given
        redshift. It is simply c * lookback_time.
        """
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.lookback_distance(self.value)

    lookback_distance = property(_calculate_lookback_distance)

    def _calculate_comoving_distance(self, cosmology=None):
        """Comoving line-of-sight distance in Mpc at each input redshift."""
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.comoving_distance(self.value)

    comoving_distance = property(_calculate_comoving_distance)

    def _calculate_luminosity_distance(self, cosmology=None):
        """Luminosity distance in Mpc at each input redshift."""
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.luminosity_distance(self.value)

    luminosity_distance = property(_calculate_luminosity_distance)

    def _calculate_distance_modulus(self, cosmology=None):
        """Distance in Mag at each input redshift.

        The distance modulus is defined as the (apparent magnitude -
        absolute magnitude) for an object at redshift ``z``.
        """
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.distmod(self.value)

    distance_modulus = property(_calculate_distance_modulus)

    def _calculate_temperature(self, cosmology=None):
        """The CMB temperature in K at each input redshift."""
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.Tcmb(self.value)

    temperature = property(_calculate_temperature)

    def _calculate_age(self, cosmology=None):
        """The age in Gyr at each input redshift."""
        cosmology = self._resolve_cosmology(cosmology)
        return cosmology.age(self.value)

    age = property(_calculate_age)
