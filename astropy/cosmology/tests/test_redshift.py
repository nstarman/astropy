# Licensed under a 3-clause BSD style license - see LICENSE.rst

import io
import sys
from contextlib import nullcontext

import pytest

import numpy as np

from astropy import units as u
from astropy.cosmology import WMAP5, default_cosmology
from astropy.cosmology.redshift import CosmologicalRedshift
from astropy.units import allclose
from astropy.utils.compat.optional_deps import HAS_SCIPY
from astropy.utils.exceptions import AstropyUserWarning

cosmologies = pytest.mark.parametrize("cosmology", [None, WMAP5])
redshift_names = pytest.mark.parametrize("name", ["z0", "zinf", "z1100", "zs", "znpy"])


################################################################################


@pytest.mark.skipif(not HAS_SCIPY, reason="Needs Scipy.")
class TestCosmologicalRedshift:
    def setup_class(self):

        self.z0 = CosmologicalRedshift(0)
        self.zinf = CosmologicalRedshift(np.inf)
        self.z1100 = CosmologicalRedshift(1100)
        self.zs = CosmologicalRedshift(np.linspace(0, 1100, num=11))
        self.znpy = CosmologicalRedshift(list(np.linspace(0, 1100, num=11)))

    # ===============================================================
    # Attribute and Method Tests

    @redshift_names
    def test_dimensionless(self, name):
        """Test attribute ``CosmologicalRedshift.scale_factor``."""
        z = getattr(self, name)

        assert np.all(z.dimensionless.value == z.value)
        assert z.dimensionless.unit == u.one

    # ================================================

    @cosmologies
    @redshift_names
    def test_scale_factor(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.scale_factor``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.scale_factor(z)

        with default_cosmology.set(cosmology):
            assert np.all(z.scale_factor == expected)

    @cosmologies
    @redshift_names
    def test_lookback_distance(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.lookback_distance``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.lookback_distance(z if z.isscalar else list(z))

        with default_cosmology.set(cosmology):
            assert np.all(z.lookback_distance == expected)

    @cosmologies
    @redshift_names
    def test_comoving_distance(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.comoving_distance``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.comoving_distance(z if z.isscalar else list(z))

        with default_cosmology.set(cosmology):
            assert np.all(z.comoving_distance == expected)

    @cosmologies
    @redshift_names
    def test_luminosity_distance(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.luminosity_distance``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.luminosity_distance(z if z.isscalar else list(z))

        with default_cosmology.set(cosmology):
            assert np.all(z.luminosity_distance == expected)

    @cosmologies
    @redshift_names
    def test_distance_modulus(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.distance_modulus``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology

        with (
            pytest.warns(RuntimeWarning, match="zero")
            if np.any(z == 0)
            else nullcontext()
        ):  # acknowledge then ignore zero-division warning.
            expected = cosmology.distmod(z if z.isscalar else list(z))

            with default_cosmology.set(cosmology):
                assert np.all(z.distance_modulus == expected)

    @cosmologies
    @redshift_names
    def test_temperature(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.temperature``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.Tcmb(z if z.isscalar else list(z))

        with default_cosmology.set(cosmology):
            assert np.all(z.temperature == expected)

    @cosmologies
    @redshift_names
    def test_age(self, name, cosmology):
        """Test attribute ``CosmologicalRedshift.age``."""
        z = getattr(self, name)
        cosmology = default_cosmology.get() if cosmology is None else cosmology
        expected = cosmology.age(z if z.isscalar else list(z))

        with default_cosmology.set(cosmology):
            assert np.all(z.age == expected)

    # ===============================================================
    # Usage Tests

    @redshift_names
    def test_construction(self, name):
        """Test the various means of constructing a CosmologicalRedshift."""
        z = getattr(self, name).value

        # Redshift inputs
        zr = CosmologicalRedshift(z)
        assert np.all(zr.value == z)

        # Scale factor
        cosmo = default_cosmology.get()
        with (
            pytest.warns(RuntimeWarning, match="zero")
            if np.any(z == np.inf)
            else nullcontext()
        ):  # acknowledge then ignore zero-division warning.
            za = CosmologicalRedshift(scale_factor=cosmo.scale_factor(z))
            assert np.allclose(za.value, z)

        # 0 inputs
        with pytest.raises(ValueError, match="constructor"):
            CosmologicalRedshift()

        # 2 inputs
        with pytest.raises(ValueError, match="Values were given for"):
            CosmologicalRedshift(z, scale_factor=cosmo.scale_factor(z))
