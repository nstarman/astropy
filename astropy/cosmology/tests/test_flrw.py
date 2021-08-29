# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Testing :mod:`astropy.cosmology.flrw`."""

##############################################################################
# IMPORTS

import abc
import copy

import pytest

import numpy as np

import astropy.constants as const
import astropy.units as u
from astropy.cosmology import (FLRW, FlatLambdaCDM, Flatw0waCDM, FlatwCDM,
                               LambdaCDM, w0waCDM, w0wzCDM, wCDM, wpwaCDM)
from astropy.cosmology.core import _COSMOLOGY_CLASSES
from astropy.units import allclose

from .test_core import TestCosmology as CosmologyTest

##############################################################################
# TESTS
##############################################################################


class TestFLRW(CosmologyTest):

    props = ["H0", "Om0", "Ode0", "Ob0", "Odm0", "Ok0", "Tcmb0", "Tnu0", "Neff", "Ogamma0"]
    props += ["h", "hubble_time", "hubble_distance", "critical_density0", "Onu0"]

    def setup_class(self):
        """
        Setup for testing.
        FLRW is abstract, so tests are done on a subclass.
        """

        class SubFLRW(FLRW):
            def w(self, z):
                return super().w(z)

        self.cls = SubFLRW
        # H0, Om0, Ode0
        self.cls_args = (70 * u.km / u.s / u.Mpc, 0.27 * u.one, 0.689 * u.one)
        self.cls_kwargs = dict(Tcmb0=3.0 * u.K, name="test", meta={"a": "b"})

    def cleanup_class(self):
        _COSMOLOGY_CLASSES.pop("TestFLRW.setup_class.<locals>.SubFLRW")

    # ===============================================================
    # tests modified from parent class

    def test_clone(self, cosmo, cosmo_class):
        super().test_clone(cosmo)

        # Now change H0
        # Note that H0 affects Ode0 because it changes Ogamma0
        newclone = cosmo.clone(H0=60 * u.km / u.s / u.Mpc)
        assert newclone is not cosmo
        assert newclone.__class__ == cosmo.__class__
        assert newclone.name == cosmo.name + " (modified)"
        assert not allclose(newclone.H0.value, cosmo.H0.value)
        assert allclose(newclone.H0, 60.0 * u.km / u.s / u.Mpc)
        assert allclose(newclone.Om0, cosmo.Om0)
        assert allclose(newclone.Ok0, cosmo.Ok0)
        assert not allclose(newclone.Ogamma0, cosmo.Ogamma0)
        assert not allclose(newclone.Onu0, cosmo.Onu0)
        assert allclose(newclone.Tcmb0, cosmo.Tcmb0)
        assert allclose(newclone.m_nu, cosmo.m_nu)
        assert allclose(newclone.Neff, cosmo.Neff)

        z = np.linspace(0.1, 3, 15)

        # Compare modified version with directly instantiated one
        arguments = copy.deepcopy(newclone._init_arguments)
        ba = newclone._init_signature.bind(**arguments)
        cmp = cosmo_class(*ba.args, **ba.kwargs)
        assert newclone.__class__ == cmp.__class__
        assert cmp.name == newclone.name
        assert allclose(newclone.H0, cmp.H0)
        assert allclose(newclone.Om0, cmp.Om0)
        assert allclose(newclone.Ode0, cmp.Ode0)
        assert allclose(newclone.Ok0, cmp.Ok0)
        assert allclose(newclone.Ogamma0, cmp.Ogamma0)
        assert allclose(newclone.Onu0, cmp.Onu0)
        assert allclose(newclone.Tcmb0, cmp.Tcmb0)
        assert allclose(newclone.m_nu, cmp.m_nu)
        assert allclose(newclone.Neff, cmp.Neff)
        assert allclose(newclone.Om(z), cmp.Om(z))
        assert allclose(newclone.H(z), cmp.H(z))
        assert allclose(newclone.luminosity_distance(z),
                        cmp.luminosity_distance(z))

        # Now try changing multiple things
        newclone = cosmo.clone(name="New name", H0=65 * u.km / u.s / u.Mpc,
                               Tcmb0=2.8 * u.K, meta=dict(zz="tops"))
        assert newclone.__class__ == cosmo.__class__
        assert not newclone.name == cosmo.name
        assert not allclose(newclone.H0.value, cosmo.H0.value)
        assert allclose(newclone.H0, 65.0 * u.km / u.s / u.Mpc)
        assert allclose(newclone.Om0, cosmo.Om0)
        assert allclose(newclone.Ok0, cosmo.Ok0)
        assert not allclose(newclone.Ogamma0, cosmo.Ogamma0)
        assert not allclose(newclone.Onu0, cosmo.Onu0)
        assert not allclose(newclone.Tcmb0.value, cosmo.Tcmb0.value)
        assert allclose(newclone.Tcmb0, 2.8 * u.K)
        assert allclose(newclone.m_nu, cosmo.m_nu)
        assert allclose(newclone.Neff, cosmo.Neff)
        assert newclone.meta == dict(a="b", zz="tops")

        # And direct comparison
        arguments = copy.deepcopy(newclone._init_arguments)
        ba = newclone._init_signature.bind(**arguments)
        cmp = cosmo_class(*ba.args, **ba.kwargs)
        assert newclone.__class__ == cmp.__class__
        assert newclone.name == cmp.name
        assert allclose(newclone.H0, cmp.H0)
        assert allclose(newclone.Om0, cmp.Om0)
        assert allclose(newclone.Ode0, cmp.Ode0)
        assert allclose(newclone.Ok0, cmp.Ok0)
        assert allclose(newclone.Ogamma0, cmp.Ogamma0)
        assert allclose(newclone.Onu0, cmp.Onu0)
        assert allclose(newclone.Tcmb0, cmp.Tcmb0)
        assert allclose(newclone.m_nu, cmp.m_nu)
        assert allclose(newclone.Neff, cmp.Neff)
        assert allclose(newclone.Om(z), cmp.Om(z))
        assert allclose(newclone.H(z), cmp.H(z))
        assert allclose(newclone.luminosity_distance(z),
                        cmp.luminosity_distance(z))

    def test_clone_fail_unexpected_kwarg(self, cosmo):
        """Test exception if user passes non-parameter."""
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            cosmo.clone(not_an_arg=4)

    # ---------------------------------------------------------------

    def test_readwrite_from_subclass_partial_info(self, cosmos, tmpdir, readwrite_format):
        """
        Test writing from an instance and reading from that class.
        This requires partial information.

        .. todo::

            generalize over all save formats for this test.
        """
        format = "json"
        fname = tmpdir / f"{cosmos.name}.{readwrite_format}"

        cosmos.write(str(fname), format=readwrite_format)

        # partial information
        with open(fname, "r") as file:
            L = file.readlines()[0]
        L = L[:L.index('"cosmology":')]+L[L.index(', ')+2:]  # remove cosmology
        i = L.index('"Tcmb0":')  # delete Tcmb0
        L = L[:i] + L[L.index(', ', L.index(', ', i) + 1)+2:]  # second occurence

        tempfname = tmpdir / f"{cosmos.name}_temp.{format}"
        with open(tempfname, "w") as file:
            file.writelines([L])

        # read with the same class that wrote fills in the missing info with
        # the default value
        got = cosmos.__class__.read(tempfname, format=readwrite_format)
        got2 = Cosmology.read(tempfname, format=readwrite_format, cosmology=cosmos.__class__)
        got3 = Cosmology.read(tempfname, format=readwrite_format,
                              cosmology=cosmos.__class__.__qualname__)

        assert (got == got2) and (got2 == got3)  # internal consistency

        # not equal, because Tcmb0 is changed
        assert got != cosmos
        assert got.Tcmb0 == cosmos.__class__._init_signature.parameters["Tcmb0"].default
        assert got.clone(name=cosmos.name, Tcmb0=cosmos._init_arguments["Tcmb0"]) == cosmos
        # but the metadata is the same
        assert got.meta == cosmos.meta

    def test_reader_class_mismatch(self, cosmos, tmpdir, readwrite_format):
        """Test when the reader class doesn't match the file."""
        fname = tmpdir / f"{cosmos.name}.{readwrite_format}"
        cosmos.write(str(fname), format=readwrite_format)

        # class mismatch
        # when reading directly
        with pytest.raises(TypeError, match="missing 1 required"):
            w0wzCDM.read(fname, format=readwrite_format)

        with pytest.raises(TypeError, match="missing 1 required"):
            Cosmology.read(fname, format=readwrite_format, cosmology=w0wzCDM)

        # when specifying the class
        with pytest.raises(ValueError, match="`cosmology` must be either"):
            w0wzCDM.read(fname, format=readwrite_format, cosmology="FlatLambdaCDM")

    def test_format_from_subclass_partial_info(self, cosmos):
        """
        Test writing from an instance and reading from that class.
        This requires partial information.

        .. todo::

            generalize over all formats for this test.
        """
        format, objtype = ("mapping", dict)

        # test to_format
        obj = cosmos.to_format(format)
        assert isinstance(obj, objtype)

        # partial information
        tempobj = copy.deepcopy(obj)
        del tempobj["cosmology"]
        del tempobj["Tcmb0"]

        # read with the same class that wrote fills in the missing info with
        # the default value
        got = cosmos.__class__.from_format(tempobj, format=format)
        got2 = Cosmology.from_format(tempobj, format=format, cosmology=cosmos.__class__)
        got3 = Cosmology.from_format(tempobj, format=format, cosmology=cosmos.__class__.__qualname__)

        assert (got == got2) and (got2 == got3)  # internal consistency

        # not equal, because Tcmb0 is changed
        assert got != cosmos
        assert got.Tcmb0 == cosmos.__class__._init_signature.parameters["Tcmb0"].default
        assert got.clone(name=cosmos.name, Tcmb0=cosmos.Tcmb0) == cosmos
        # but the metadata is the same
        assert got.meta == cosmos.meta

    def test_format_reader_class_mismatch(self, cosmos, tofrom_format):
        """Test when the reader class doesn't match the object."""
        format, objtype = tofrom_format

        # test to_format
        obj = cosmos.to_format(format)
        assert isinstance(obj, objtype)

        # class mismatch
        with pytest.raises(TypeError, match="missing 1 required"):
            w0wzCDM.from_format(obj, format=format)

        with pytest.raises(TypeError, match="missing 1 required"):
            Cosmology.from_format(obj, format=format, cosmology=w0wzCDM)

        # when specifying the class
        with pytest.raises(ValueError, match="`cosmology` must be either"):
            w0wzCDM.from_format(obj, format=format, cosmology="FlatLambdaCDM")

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        # super().test_init(cosmo_class)  # FLRW doesn't accept all args, kwarg
        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        assert False

    @pytest.mark.parametrize("prop", props)
    def test_parameter_property(self, cosmo, prop):
        # property test
        assert getattr(cosmo, prop) is getattr(cosmo, "_" + prop)

        # test immutability
        with pytest.raises(AttributeError):
            setattr(cosmo, prop, None)

    def test_H0(self, cosmos):
        """Test Hubble constant is same as initialization"""
        # values are equal
        assert u.Quantity(cosmos.H0).value == self.cls_args[0].value

        # and if should have units, check it does.
        if isinstance(cosmos._init_arguments["H0"], u.Quantity):
            assert isinstance(cosmos.H0, u.Quantity)
            assert cosmos.H0.unit == u.km / u.Mpc / u.s

    def test_Tcmb0(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.Tcmb0`."""
        # units
        assert cosmo.Tcmb0.unit == u.K

    def test_Tnu0(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.Tnu0`."""
        # units
        assert cosmo.Tnu0.unit == u.K

    @pytest.mark.skip("TODO!")
    def test_has_massive_nu(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_m_nu(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_h(self, cosmo):
        assert False

    def test_w(self, cosmo):
        with pytest.raises(NotImplementedError, match="not implemented"):
            cosmo.w(None)

    @pytest.mark.skip("TODO!")
    def test_Om(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Ob(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Odm(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Ok(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Ode(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Ogamma(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_Onu(self, cosmo):
        assert False

    def test_Tcmb(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.Tcmb`."""
        # units
        assert cosmo.Tcmb(1.0).unit == u.K
        assert cosmo.Tcmb([0.0, 1.0]).unit == u.K

        # cosmo = flrw.FlatLambdaCDM(70.4, 0.272, Tcmb0=2.5)
        assert allclose(cosmo.Tcmb0, 2.5 * u.K)
        assert allclose(cosmo.Tcmb(2), 7.5 * u.K)
        z = [0.0, 1.0, 2.0, 3.0, 9.0]
        assert allclose(cosmo.Tcmb(z),
                        [2.5, 5.0, 7.5, 10.0, 25.0] * u.K, rtol=1e-6)
        # Make sure it's the same for integers
        z = [0, 1, 2, 3, 9]
        assert allclose(cosmo.Tcmb(z),
                        [2.5, 5.0, 7.5, 10.0, 25.0] * u.K, rtol=1e-6)

    def test_Tnu(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.Tnu`."""
        # units
        assert cosmo.Tnu(1.0).unit == u.K
        assert cosmo.Tnu([0.0, 1.0]).unit == u.K

        # cosmo = flrw.FlatLambdaCDM(70.4, 0.272, Tcmb0=3.0)
        assert allclose(cosmo.Tnu0, 2.1412975665108247 * u.K, rtol=1e-6)
        assert allclose(cosmo.Tnu(2), 6.423892699532474 * u.K, rtol=1e-6)
        z = [0.0, 1.0, 2.0, 3.0]
        expected = [2.14129757, 4.28259513, 6.4238927, 8.56519027] * u.K
        assert allclose(cosmo.Tnu(z), expected, rtol=1e-6)
    
        # Test for integers
        z = [0, 1, 2, 3]
        assert allclose(cosmo.Tnu(z), expected, rtol=1e-6)

    @pytest.mark.skip("TODO!")
    def test_nu_relative_density(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__w_integrand(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_efunc(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_inv_efunc(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__lookback_time_integrand_scalar(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_lookback_time_integrand(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__abs_distance_integrand_scalar(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_abs_distance_integrand(self, cosmo):
        assert False

    def test_H(self, cosmo):
        """Test Hubble parameter depends on w(z)."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.H(1.0)

    @pytest.mark.skip("TODO!")
    def test_scale_factor(self, cosmo):
        assert False

    def test_lookback_time(self, cosmo):
        """Test :meth:`astropy.cosmology.lookback_time`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.lookback_time(1.0)

    @pytest.mark.skip("TODO!")
    def test__lookback_time(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__integral_lookback_time(self, cosmo):
        assert False

    def test_lookback_distance(self):
        """Test :meth:`astropy.cosmology.lookback_distance`."""
        # test units
        assert cosmo.lookback_distance(1.0).unit == u.Mpc

    def test_age(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.age`."""
        # units
        assert cosmo.age(1.0).unit == u.Gyr

    @pytest.mark.skip("TODO!")
    def test__age(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__integral_age(self, cosmo):
        assert False

    def test_critical_density(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.critical_density`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.critical_density(1.0)

    def test_comoving_distance(self, cosmo):
        """Test :meth:`astropy.cosmology.comoving_distance`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.comoving_distance(1.0)

    def test__comoving_distance_z1z2(self, cosmo):
        """Test :meth:`astropy.cosmology._comoving_distance_z1z2`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo._comoving_distance_z1z2(1.0, 2.0)

    @pytest.mark.skip("TODO!")
    def test__integral_comoving_distance_z1z2(self, cosmo):
        assert False

    def test_comoving_transverse_distance(self):
        """Test :meth:`astropy.cosmology.comoving_transverse_distance`."""

    def test__comoving_transverse_distance_z1z2(self):
        """Test :meth:`astropy.cosmology._comoving_transverse_distance_z1z2`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo._comoving_transverse_distance_z1z2(1.0, 2.0)

    def test_angular_diameter_distance(self):
        """Test :meth:`astropy.cosmology.angular_diameter_distance`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.angular_diameter_distance(1.0)

    def test_luminosity_distance(self):
        """Test :meth:`astropy.cosmology.luminosity_distance`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.luminosity_distance(1.0)

    def test_angular_diameter_distance_z1z2(self):
        """Test :meth:`astropy.cosmology.angular_diameter_distance_z1z2`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.angular_diameter_distance_z1z2(1.0, 2.0)

    @pytest.mark.skip("TODO!")
    def test_absorption_distance(self, cosmo):
        assert False

    def test_distmod(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.distmod`."""
        # units
        assert cosmo.distmod(1.0).unit == u.mag

    @pytest.mark.skip("TODO!")
    def test_comoving_volume(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.comoving_volume`."""
        # units
        assert cosmo.comoving_volume(1.0).unit == u.Mpc ** 3

    @pytest.mark.skip("TODO!")
    def test_differential_comoving_volume(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_kpc_comoving_per_arcmin(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.kpc_comoving_per_arcmin`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.kpc_comoving_per_arcmin(1.0)

    @pytest.mark.skip("TODO!")
    def test_kpc_proper_per_arcmin(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.kpc_proper_per_arcmin`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.kpc_proper_per_arcmin(1.0)

    @pytest.mark.skip("TODO!")
    def test_arcsec_per_kpc_comoving(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.arcsec_per_kpc_comoving`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.arcsec_per_kpc_comoving(1.0)

    @pytest.mark.skip("TODO!")
    def test_arcsec_per_kpc_proper(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.arcsec_per_kpc_proper`."""
        with pytest.raises(NotImplementedError, match="not implemented"):
            assert cosmo.arcsec_per_kpc_proper(1.0)


# ----------------------------------------------------------------------------


class FLRWBaseTest(TestFLRW, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def test_w(self, cosmo):
        pass

    @abc.abstractmethod
    def test_distance(self):
        """
        Test distance calculations for various special case scenarios (no
        relativistic species, normal, massive neutrinos) These do not come from
        external codes -- they are just internal checks to make sure nothing
        changes if we muck with the distance calculators.
        """

    # ===============================================================
    # reliant on non-abstract w(z)

    def test_H(self, cosmo):
        """Test Hubble parameter."""
        # test units
        assert cosmo.H(1.0).unit == u.km / u.Mpc / u.s

    def test_lookback_time(self, cosmo):
        """Test :meth:`astropy.cosmology.lookback_time`."""
        # test units
        assert cosmo.lookback_time(1.0).unit == u.Gyr

    def test_critical_density(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.critical_density`."""
        # units
        assert cosmo.critical_density(1.0).unit == u.g / u.cm ** 3

    def test_comoving_distance(self, cosmo):
        """Test :meth:`astropy.cosmology.comoving_distance`."""
        # test units
        assert cosmo.comoving_distance(1.0).unit == u.Mpc

    def test__comoving_distance_z1z2(self, cosmo):
        """Test :meth:`astropy.cosmology._comoving_distance_z1z2`."""
        # test units
        assert cosmo._comoving_distance_z1z2(1.0, 2.0).unit == u.Mpc

    def test_comoving_transverse_distance(self, cosmo):
        """Test :meth:`astropy.cosmology.comoving_transverse_distance`."""
        # test units
        assert cosmo.comoving_transverse_distance(1.0).unit == u.Mpc

    def test__comoving_transverse_distance_z1z2(self, cosmo):
        """Test :meth:`astropy.cosmology._comoving_transverse_distance_z1z2`."""
        # test units
        assert cosmo._comoving_transverse_distance_z1z2(1.0, 2.0).unit == u.Mpc

    def test_angular_diameter_distance(self, cosmo):
        """Test :meth:`astropy.cosmology.angular_diameter_distance`."""
        # test units
        assert cosmo.angular_diameter_distance(1.0).unit == u.Mpc

    def test_luminosity_distance(self, cosmo):
        """Test :meth:`astropy.cosmology.luminosity_distance`."""
        # test units
        assert cosmo.luminosity_distance(1.0).unit == u.Mpc

    def test_angular_diameter_distance_z1z2(self, cosmo):
        """Test :meth:`astropy.cosmology.angular_diameter_distance_z1z2`."""
        # test units
        assert cosmo.angular_diameter_distance_z1z2(1.0, 2.0).unit == u.Mpc

    def test_kpc_comoving_per_arcmin(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.kpc_comoving_per_arcmin`."""
        # units
        assert cosmo.kpc_comoving_per_arcmin(1.0).unit == u.kpc / u.arcmin

    def test_kpc_proper_per_arcmin(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.kpc_proper_per_arcmin`."""
        # units
        assert cosmo.kpc_proper_per_arcmin(1.0).unit == u.kpc / u.arcmin

    def test_arcsec_per_kpc_comoving(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.arcsec_per_kpc_comoving`."""
        # units
        assert cosmo.arcsec_per_kpc_comoving(1.0).unit == u.arcsec / u.kpc

    def test_arcsec_per_kpc_proper(self, cosmo):
        """Test :meth:`astropy.cosmology.FLRW.arcsec_per_kpc_proper`."""
        # units
        assert cosmo.arcsec_per_kpc_proper(1.0).unit == u.arcsec / u.kpc

    # ===============================================================

    def test_efunc_vs_invefunc(self, cosmo):
        """Test that efunc and inv_efunc give inverse values.

        Note that the test doen't need scipy because it doesn't need to call
        ``de_density_scale``.
        """
        z0 = 0.5
        z = np.array([0.5, 1.0, 2.0, 5.0])

        assert allclose(cosmo.efunc(z0), 1.0 / cosmo.inv_efunc(z0))
        assert allclose(cosmo.efunc(z), 1.0 / cosmo.inv_efunc(z))


class FlatCosmologyTestMixin:
    """Mixin tests for a flat cosmology."""

    def test_Ode0_derived(self, cosmos):
        """Test ``0de0`` is set by 1 - everything else."""
        # should add to 1
        assert allclose(cosmos.Ode0, 1.0 - cosmos.Om0 - cosmos.Ogamma0 - cosmos.Onu0, rtol=1e-6)

    def test_curvature(self, cosmos):
        """Test no curvature."""
        assert cosmos.Ok0 == 0.0

    def test_Odm0(self, cosmo):
        """Test ``Odm0 = Om0 - Ob0``."""
        # Ob0 can be None
        if cosmo.Ob0 is None:
            assert cosmo.Odm0 is None

        # an explicit test
        cosmo2 = cosmo.clone(Ob0=0.05)
        assert allclose(cosmo2.Odm0, cosmo2.Om0 - 0.05)

    def test_evolved_density_sums_to_one(self, cosmos):
        assert allclose(
            cosmos.Om(1) + cosmos.Ode(1) + cosmos.Ogamma(1) + cosmos.Onu(1), 1.0, rtol=1e-6
        )


# ----------------------------------------------------------------------------


class TestLambdaCDM(FLRWBaseTest):
    def setup_class(self):
        """Setup for testing."""
        self.cls = LambdaCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27, 0.73)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_w(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__elliptic_comoving_distance_z1z2(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__dS_comoving_distance_z1z2(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__EdS_comoving_distance_z1z2(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__hypergeometric_comoving_distance_z1z2(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__T_hypergeometric(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__dS_age(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__EdS_age(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__flat_age(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__EdS_lookback_time(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__dS_lookback_time(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test__flat_lookback_time(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_efunc(self, cosmo):
        assert False

    @pytest.mark.skip("TODO!")
    def test_inv_efunc(self, cosmo):
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])
    
        # The pattern here is: no relativistic species, the relativistic
        # species with massless neutrinos, then massive neutrinos
        cos = cosmo_class(75.0, 0.25, 0.5, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [2953.93001902, 4616.7134253, 5685.07765971,
                         6440.80611897] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.6, Tcmb0=3.0, Neff=3,
                             m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [3037.12620424, 4776.86236327, 5889.55164479,
                         6671.85418235] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.3, 0.4, Tcmb0=3.0, Neff=3,
                             m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2471.80626824, 3567.1902565, 4207.15995626,
                         4638.20476018] * u.Mpc, rtol=1e-4)

    def test_elliptic_comoving_distance_z1z2(self):
        """Regression test for #8388."""
        cosmo = LambdaCDM(70., 2.3, 0.05, Tcmb0=0)
        z = 0.2
        assert allclose(cosmo.comoving_distance(z),
                        cosmo._integral_comoving_distance_z1z2(0., z))
        assert allclose(cosmo._elliptic_comoving_distance_z1z2(0., z),
                        cosmo._integral_comoving_distance_z1z2(0., z))

    # Elliptic cosmology
    @pytest.mark.parametrize('cosmo', [LambdaCDM(H0=70, Om0=0.3, Ode0=0.6, Tcmb0=0.0)])
    @pytest.mark.parametrize('z', [
        (0, 1, 2, 3, 4),  # tuple
        [0, 1, 2, 3, 4],  # list
        np.array([0, 1, 2, 3, 4]),  # array
    ])
    def test_comoving_distance_iterable_argument(self, cosmo, z):
        """
        Regression test for #10980
        Test that specialized comoving distance methods handle iterable arguments.
        """
        assert allclose(cosmo.comoving_distance(z),
                        cosmo._integral_comoving_distance_z1z2(0., z))

    # Elliptic cosmology
    @pytest.mark.parametrize('cosmo', [LambdaCDM(H0=70, Om0=0.3, Ode0=0.6, Tcmb0=0.0)])
    def test_comoving_distance_broadcast(self, cosmo):
        """
        Regression test for #10980
        Test that specialized comoving distance methods broadcast array arguments.
        """
        z1 = np.zeros((2, 5))
        z2 = np.ones((3, 1, 5))
        z3 = np.ones((7, 5))
        output_shape = np.broadcast(z1, z2).shape

        # Check compatible array arguments return an array with the correct shape
        assert cosmo._comoving_distance_z1z2(z1, z2).shape == output_shape

        # Check incompatible array arguments raise an error
        with pytest.raises(ValueError, match='z1 and z2 have different shapes'):
            cosmo._comoving_distance_z1z2(z1, z3)


# ----------------------------------------------------------------------------


class TestFlatLambdaCDM(FlatCosmologyTestMixin, TestLambdaCDM):
    def setup_class(self):
        """Setup for testing."""
        self.cls = FlatLambdaCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

    @pytest.mark.parametrize(
        "kwargs, match",
        [
            (dict(Om0=-0.27), "Matter density can not be negative"),
            (dict(Neff=-1), "Effective number of neutrinos can not be negative"),
            (dict(Tcmb0=u.Quantity([0.0, 2], u.K)), "Tcmb0 is a non-scalar"),
            (dict(H0=[70, 100] * u.km / u.s / u.Mpc, Om0=0.27), "H0 is a non-scalar"),
            (dict(Om0=0.2, Tcmb0=3, m_nu=[-0.3, 0.2, 0.1] * u.eV), "Invalid (.*?) neutrino"),
            (dict(Om0=0.2, Tcmb0=3, Neff=2, m_nu=[0.15, 0.2, 0.1] * u.eV), "number of neutrino"),
            (dict(Om0=0.2, Tcmb0=3, m_nu=[-0.3, 0.2] * u.eV), "Invalid (.*?) neutrino"),
            (dict(Ob0=-0.04), "Baryonic density can not be negative"),
            (dict(Ob0=0.4), "Baryonic density can not be larger than total matter density"),
        ],
    )
    def test_init_error(self, cosmo_class, kwargs, match):
        kwargs.setdefault("H0", self.cls_args[0])
        kwargs.setdefault("Om0", self.cls_args[1])
    
        with pytest.raises(ValueError, match=match):
            cosmo = cosmo_class(**kwargs)

    # -------------------------------------------

    def test_absorption_distance(self, cosmo_class):
        """Test ``absorption_distance``. TODO! generalize test to FLRW."""
        cosmo = cosmo_class(70.4, 0.272, Tcmb0=0.0)
        assert allclose(cosmo.absorption_distance([1, 3]),
                        [1.72576635, 7.98685853])
        assert allclose(cosmo.absorption_distance([1., 3.]),
                        [1.72576635, 7.98685853])
        assert allclose(cosmo.absorption_distance(3), 7.98685853)
        assert allclose(cosmo.absorption_distance(3.), 7.98685853)

    def test_Ob_error(self, cosmo_class):
        with pytest.raises(ValueError):
            cosmo = cosmo_class(*self.cls_args)
            cosmo.Ob(1)

    def test_Odm_error(self, cosmo_class):
        with pytest.raises(ValueError):
            cosmo = cosmo_class(*self.cls_args)
            cosmo.Odm(1)

    @pytest.mark.skip("TODO!")
    def test_efunc(self, cosmo):
        super().test_efunc(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_inv_efunc(self, cosmo):
        super().test_inv_efunc(cosmo)
        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================
    
    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        # Flat
        cos = cosmo_class(75.0, 0.25, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [3180.83488552, 5060.82054204, 6253.6721173,
                         7083.5374303] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0, Neff=3,
                                  m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [3180.42662867, 5059.60529655, 6251.62766102,
                         7080.71698117] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0, Neff=3,
                                  m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2337.54183142, 3371.91131264, 3988.40711188,
                         4409.09346922] * u.Mpc, rtol=1e-4)

        # Also test different numbers of massive neutrinos
        # for cosmo_class to give the scalar nu density functions a
        # work out
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0,
                                 m_nu=u.Quantity([10.0, 0, 0], u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2777.71589173, 4186.91111666, 5046.0300719,
                         5636.10397302] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0,
                                 m_nu=u.Quantity([10.0, 5, 0], u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2636.48149391, 3913.14102091, 4684.59108974,
                         5213.07557084] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0,
                                 m_nu=u.Quantity([4.0, 5, 9], u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2563.5093049, 3776.63362071, 4506.83448243,
                         5006.50158829] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, Tcmb0=3.0, Neff=4.2,
                                 m_nu=u.Quantity([1.0, 4.0, 5, 9], u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2525.58017482, 3706.87633298, 4416.58398847,
                         4901.96669755] * u.Mpc, rtol=1e-4)

    @pytest.mark.parametrize('cosmo', [
        FlatLambdaCDM(H0=70, Om0=0.0, Tcmb0=0.0),  # de Sitter
        FlatLambdaCDM(H0=70, Om0=1.0, Tcmb0=0.0),  # Einstein - de Sitter
        FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0.0),  # Hypergeometric
    ])
    @pytest.mark.parametrize('z', [
        (0, 1, 2, 3, 4),  # tuple
        [0, 1, 2, 3, 4],  # list
        np.array([0, 1, 2, 3, 4]),  # array
    ])
    def test_comoving_distance_iterable_argument(cosmo, z):
        super().test_comoving_distance_iterable_argument(cosmo, z)

    @pytest.mark.parametrize('cosmo', [
        FlatLambdaCDM(H0=70, Om0=0.0, Tcmb0=0.0),  # de Sitter
        FlatLambdaCDM(H0=70, Om0=1.0, Tcmb0=0.0),  # Einstein - de Sitter
        FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=0.0),  # Hypergeometric
    ])
    def test_comoving_distance_broadcast(cosmo):
        """
        Regression test for #10980
        Test that specialized comoving distance methods broadcast array arguments.
        """
        super().test_comoving_distance_broadcast(cosmo)


# ----------------------------------------------------------------------------


class TestwCDM(FLRWBaseTest):

    props = FLRWBaseTest.props + ["w0"]

    def setup_class(self):
        """Setup for testing."""
        self.cls = wCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27, 0.73)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_w(self, cosmo):
        assert False

    # TODO! better inheritance
    @pytest.mark.parametrize("prop", props)
    def test_parameter_property(self, cosmo, prop):
        super().test_parameter_property(cosmo, prop)

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        super().test_de_density_scale(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_efunc(self, cosmo):
        super().test_efunc(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_inv_efunc(self, cosmo):
        super().test_inv_efunc(cosmo)
        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================
    
    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        # Non-flat w
        cos = cosmo_class(75.0, 0.25, 0.4, w0=-0.9, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [2849.6163356, 4428.71661565, 5450.97862778,
                         6179.37072324] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.4, w0=-1.1, Tcmb0=3.0, Neff=3,
                          m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2904.35580229, 4511.11471267, 5543.43643353,
                         6275.9206788] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.4, w0=-0.9, Tcmb0=3.0, Neff=3,
                          m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2473.32522734, 3581.54519631, 4232.41674426,
                         4671.83818117] * u.Mpc, rtol=1e-4)

# ----------------------------------------------------------------------------


class TestFlatwCDM(FlatCosmologyTestMixin, TestwCDM):
    def setup_class(self):
        """Setup for testing."""
        self.cls = FlatwCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27)  # H0, Om0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_efunc(self, cosmo):
        super().test_efunc(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_inv_efunc(self, cosmo):
        super().test_inv_efunc(cosmo)
        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        # Add w
        cos = cosmo_class(75.0, 0.25, w0=-1.05, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [3216.8296894, 5117.2097601, 6317.05995437,
                         7149.68648536] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, w0=-0.95, Tcmb0=3.0, Neff=3,
                          m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [3143.56537758, 5000.32196494, 6184.11444601,
                         7009.80166062] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, w0=-0.9, Tcmb0=3.0, Neff=3,
                          m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2337.76035371, 3372.1971387, 3988.71362289,
                         4409.40817174] * u.Mpc, rtol=1e-4)


# ----------------------------------------------------------------------------


class Testw0waCDM(FLRWBaseTest):
    """Test :class:`astropy.cosmology.w0waCDM`."""

    props = FLRWBaseTest.props + ["w0", "wa"]

    def setup_class(self):
        """Setup for testing."""
        self.cls = w0waCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27, 0.73)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================

    # TODO! generalize to apply to each change from FLRW by a subclass
    def test_clone(self, cosmo, cosmo_class):
        """Try a dark energy class, make sure it can handle w params"""
        super().test_clone(cosmo)

        cosmo = cosmo_class(name="test w0wa", H0=70 * u.km / u.s / u.Mpc,
                            Om0=0.27, Ode0=0.5, wa=0.1, Tcmb0=4.0 * u.K)
        newclone = cosmo.clone(w0=-1.1, wa=0.2)
        assert newclone.__class__ == cosmo.__class__
        assert newclone.name == cosmo.name + " (modified)"
        assert allclose(newclone.H0, cosmo.H0)
        assert allclose(newclone.Om0, cosmo.Om0)
        assert allclose(newclone.Ode0, cosmo.Ode0)
        assert allclose(newclone.Ok0, cosmo.Ok0)
        assert not allclose(newclone.w0, cosmo.w0)
        assert allclose(newclone.w0, -1.1)
        assert not allclose(newclone.wa, cosmo.wa)
        assert allclose(newclone.wa, 0.2)

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_w(self, cosmo):
        assert False

    # TODO! better inheritance
    @pytest.mark.parametrize("prop", props)
    def test_parameter_property(self, cosmo, prop):
        super().test_parameter_property(cosmo, prop)

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        super().test_de_density_scale(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        # w0wa
        cos = cosmo_class(75.0, 0.3, 0.6, w0=-0.9, wa=0.1, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [2937.7807638, 4572.59950903, 5611.52821924,
                         6339.8549956] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, w0=-0.9, wa=0.1, Tcmb0=3.0, Neff=3,
                           m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2907.34722624, 4539.01723198, 5593.51611281,
                         6342.3228444] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, w0=-0.9, wa=0.1, Tcmb0=3.0, Neff=3,
                           m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2507.18336722, 3633.33231695, 4292.44746919,
                         4736.35404638] * u.Mpc, rtol=1e-4)

# ----------------------------------------------------------------------------


class TestFlatw0waCDM(FlatCosmologyTestMixin, Testw0waCDM):
    def setup_class(self):
        """Setup for testing."""
        self.cls = Flatw0waCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27)  # H0, Om0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        cos = cosmo_class(75.0, 0.25, w0=-0.95, wa=0.15, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [3123.29892781, 4956.15204302, 6128.15563818,
                         6948.26480378] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, w0=-0.95, wa=0.15, Tcmb0=3.0, Neff=3,
                               m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [3122.92671907, 4955.03768936, 6126.25719576,
                         6945.61856513] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, w0=-0.95, wa=0.15, Tcmb0=3.0, Neff=3,
                               m_nu=u.Quantity(10.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2337.70072701, 3372.13719963, 3988.6571093,
                         4409.35399673] * u.Mpc, rtol=1e-4)


# ----------------------------------------------------------------------------


class TestwpwaCDM(FLRWBaseTest):
    """Test :class:`astropy.cosmology.wpwaCDM`."""

    props = FLRWBaseTest.props + ["wp", "wa", "zp"]

    def setup_class(self):
        """Setup for testing."""
        self.cls = wpwaCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27, 0.73)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_w(self, cosmo):
        assert False

    # TODO! better inheritance
    @pytest.mark.parametrize("prop", props)
    def test_parameter_property(self, cosmo, prop):
        super().test_parameter_property(cosmo, prop)

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        super().test_de_density_scale(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        cos = cosmo_class(75.0, 0.3, 0.6, wp=-0.9, zp=0.5, wa=0.1, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [2954.68975298, 4599.83254834, 5643.04013201,
                         6373.36147627] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, wp=-0.9, zp=0.4, wa=0.1,
                           Tcmb0=3.0, Neff=3, m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2919.00656215, 4558.0218123, 5615.73412391,
                         6366.10224229] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, wp=-0.9, zp=1.0, wa=0.1, Tcmb0=3.0,
                           Neff=4, m_nu=u.Quantity(5.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2629.48489827, 3874.13392319, 4614.31562397,
                         5116.51184842] * u.Mpc, rtol=1e-4)


# ----------------------------------------------------------------------------


class Testw0wzCDM(FLRWBaseTest):
    """Test :class:`astropy.cosmology.w0wzCDM`."""

    props = FLRWBaseTest.props + ["w0", "wz"]

    def setup_class(self):
        """Setup for testing."""
        self.cls = w0wzCDM
        self.cls_args = (70 * (u.km / u.s / u.Mpc), 0.27, 0.73)  # H0, Om0, Ode0
        self.cls_kwargs = dict(Tcmb0=3 * u.K, name="test", meta={"a": "b"})

    # ===============================================================
    # Method & Attribute Tests

    @pytest.mark.skip("TODO!")
    def test_init(self, cosmo_class):
        super().test_init(cosmo_class)

        assert False

    @pytest.mark.skip("TODO!")
    def test_w(self, cosmo):
        assert False

    # TODO! better inheritance
    @pytest.mark.parametrize("prop", props)
    def test_parameter_property(self, cosmo, prop):
        super().test_parameter_property(cosmo, prop)

    @pytest.mark.skip("TODO!")
    def test_de_density_scale(self, cosmo):
        super().test_de_density_scale(cosmo)

        assert False

    @pytest.mark.skip("TODO!")
    def test_repr(self, cosmos):
        super().test_repr(cosmos)
        assert False

    # ===============================================================

    def test_distance(self, cosmo_class):
        z = np.array([1.0, 2.0, 3.0, 4.0])

        cos = cosmo_class(75.0, 0.3, 0.6, w0=-0.9, wz=0.1, Tcmb0=0.0)
        assert allclose(cos.comoving_distance(z),
                        [3051.68786716, 4756.17714818, 5822.38084257,
                         6562.70873734] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, w0=-0.9, wz=0.1,
                           Tcmb0=3.0, Neff=3, m_nu=u.Quantity(0.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2997.8115653, 4686.45599916, 5764.54388557,
                         6524.17408738] * u.Mpc, rtol=1e-4)
        cos = cosmo_class(75.0, 0.25, 0.5, w0=-0.9, wz=0.1, Tcmb0=3.0,
                           Neff=4, m_nu=u.Quantity(5.0, u.eV))
        assert allclose(cos.comoving_distance(z),
                        [2676.73467639, 3940.57967585, 4686.90810278,
                         5191.54178243] * u.Mpc, rtol=1e-4)
