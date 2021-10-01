# -*- coding: utf-8 -*-

"""Testing :mod:`astropy.cosmology.units`."""

##############################################################################
# IMPORTS

import contextlib
from types import MappingProxyType

import pytest

import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology import Planck13, default_cosmology, flrw
from astropy.tests.helper import assert_quantity_allclose
from astropy.utils.compat.optional_deps import HAS_ASDF, HAS_SCIPY
from astropy.utils.exceptions import AstropyDeprecationWarning

##############################################################################
# TESTS
##############################################################################


def test_has_expected_units():
    """
    Test that this module has the expected set of units. Some of the units are
    imported from :mod:`astropy.units`, or vice versa. Here we test presence,
    not usage. Units from :mod:`astropy.units` are tested in that module. Units
    defined in :mod:`astropy.cosmology` will be tested subsequently.
    """
    with pytest.warns(AstropyDeprecationWarning, match="`littleh`"):
        assert u.astrophys.littleh is cu.littleh


def test_has_expected_equivalencies():
    """
    Test that this module has the expected set of equivalencies. Many of the
    equivalencies are imported from :mod:`astropy.units`, so here we test
    presence, not usage. Equivalencies from :mod:`astropy.units` are tested in
    that module. Equivalencies defined in :mod:`astropy.cosmology` will be
    tested subsequently.
    """
    with pytest.warns(AstropyDeprecationWarning, match="`with_H0`"):
        assert u.equivalencies.with_H0 is cu.with_H0

# ============================================================================
# Units & Equivalencies

def test_littleh():
    H0_70 = 70 * u.km / u.s / u.Mpc
    h70dist = 70 * u.Mpc / cu.littleh

    assert_quantity_allclose(h70dist.to(u.Mpc, cu.with_H0(H0_70)), 100 * u.Mpc)

    # make sure using the default cosmology works
    cosmodist = default_cosmology.get().H0.value * u.Mpc / cu.littleh
    assert_quantity_allclose(cosmodist.to(u.Mpc, cu.with_H0()), 100 * u.Mpc)

    # Now try a luminosity scaling
    h1lum = 0.49 * u.Lsun * cu.littleh ** -2
    assert_quantity_allclose(h1lum.to(u.Lsun, cu.with_H0(H0_70)), 1 * u.Lsun)

    # And the trickiest one: magnitudes.  Using H0=10 here for the round numbers
    H0_10 = 10 * u.km / u.s / u.Mpc
    # assume the "true" magnitude M = 12.
    # Then M - 5*log_10(h)  = M + 5 = 17
    withlittlehmag = 17 * (u.mag - u.MagUnit(cu.littleh ** 2))
    assert_quantity_allclose(withlittlehmag.to(u.mag, cu.with_H0(H0_10)), 12 * u.mag)


@pytest.mark.skipif(not HAS_SCIPY, reason="Cosmology needs scipy")
def test_dimensionless_redshift():
    """Test the equivalency  ``dimensionless_redshift``."""
    z = 3 * cu.redshift
    val = 3 * u.one

    # show units not equal
    assert z.unit == cu.redshift
    assert z.unit != u.one

    # test equivalency enabled by default
    assert z == val

    # also test that it works for powers
    assert (3 * cu.redshift ** 3) == val

    # and in composite units
    assert (3 * u.km / cu.redshift ** 3) == 3 * u.km

    # test it also works as an equivalency
    with u.set_enabled_equivalencies([]):  # turn off default equivalencies
        assert z.to(u.one, equivalencies=cu.dimensionless_redshift()) == val

        with pytest.raises(ValueError):
            z.to(u.one)

    # if this fails, something is really wrong
    with u.add_enabled_equivalencies(cu.dimensionless_redshift()):
        assert z == val


@pytest.mark.skipif(not HAS_SCIPY, reason="Cosmology needs scipy")
def test_redshift_temperature():
    """Test the equivalency  ``with_redshift``."""
    cosmo = Planck13.clone(Tcmb0=3 * u.K)
    default_cosmo = default_cosmology.get()
    z = 15 * cu.redshift
    Tcmb = cosmo.Tcmb(z)

    # 1) Default (without specifying the cosmology)
    with default_cosmology.set(cosmo):
        equivalency = cu.redshift_temperature()
        assert_quantity_allclose(z.to(u.K, equivalency), Tcmb)
        assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)

    # showing the answer changes if the cosmology changes
    # this test uses the default cosmology
    equivalency = cu.redshift_temperature()
    assert_quantity_allclose(z.to(u.K, equivalency), default_cosmo.Tcmb(z))
    assert default_cosmo.Tcmb(z) != Tcmb

    # 2) Specifying the cosmology
    equivalency = cu.redshift_temperature(cosmo)
    assert_quantity_allclose(z.to(u.K, equivalency), Tcmb)
    assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)

    # Test `atzkw`
    equivalency = cu.redshift_temperature(cosmo, ztol=1e-10)
    assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)


@pytest.mark.skipif(not HAS_SCIPY, reason="Cosmology needs scipy")
class Test_with_redshift:
    @pytest.fixture
    def cosmo(self):
        return Planck13.clone(Tcmb0=3 * u.K)

    # ===========================================

    def test_cosmo_different(self, cosmo):
        default_cosmo = default_cosmology.get()
        assert default_cosmo != cosmo  # shows changing default

    def test_no_equivalency(self, cosmo):
        """Test the equivalency  ``with_redshift`` without any enabled."""
        z = 15 * cu.redshift

        equivalency = cu.with_redshift(Tcmb=False)
        assert len(equivalency) == 0

    # -------------------------------------------

    def test_temperature_off(self, cosmo):
        """Test the equivalency  ``with_redshift``."""
        default_cosmo = default_cosmology.get()
        z = 15 * cu.redshift
        Tcmb = cosmo.Tcmb(z)

        # 1) Default (without specifying the cosmology)
        with default_cosmology.set(cosmo):
            equivalency = cu.with_redshift(Tcmb=False)
            with pytest.raises(u.UnitConversionError, match="'redshift' and 'K'"):
                z.to(u.K, equivalency)

        # 2) Specifying the cosmology
        equivalency = cu.with_redshift(cosmo, Tcmb=False)
        with pytest.raises(u.UnitConversionError, match="'redshift' and 'K'"):
            z.to(u.K, equivalency)

    def test_temperature(self, cosmo):
        """Test the equivalency  ``with_redshift``."""
        default_cosmo = default_cosmology.get()
        z = 15 * cu.redshift
        Tcmb = cosmo.Tcmb(z)

        # 1) Default (without specifying the cosmology)
        with default_cosmology.set(cosmo):
            equivalency = cu.with_redshift(Tcmb=True)
            assert_quantity_allclose(z.to(u.K, equivalency), Tcmb)
            assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)

        # showing the answer changes if the cosmology changes
        # this test uses the default cosmology
        equivalency = cu.with_redshift(Tcmb=True)
        assert_quantity_allclose(z.to(u.K, equivalency), default_cosmo.Tcmb(z))
        assert default_cosmo.Tcmb(z) != Tcmb

        # 2) Specifying the cosmology
        equivalency = cu.with_redshift(cosmo, Tcmb=True)
        assert_quantity_allclose(z.to(u.K, equivalency), Tcmb)
        assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)

        # Test `atzkw`
        # this is really just a test that 'atzkw' doesn't fail
        equivalency = cu.with_redshift(cosmo, Tcmb=True, atzkw={"ztol": 1e-10})
        assert_quantity_allclose(Tcmb.to(cu.redshift, equivalency), z)


# FIXME! get "dimensionless_redshift", "with_redshift" to work in this
# they are not in ``astropy.units.equivalencies``, so the following fails
@pytest.mark.skipif(not HAS_ASDF, reason="requires ASDF")
@pytest.mark.parametrize("equiv", [cu.with_H0])
def test_equivalencies_asdf(tmpdir, equiv):
    from asdf.tests import helpers

    tree = {"equiv": equiv()}
    with (
        pytest.warns(AstropyDeprecationWarning, match="`with_H0`")
        if equiv.__name__ == "with_H0"
        else contextlib.nullcontext()
    ):

        helpers.assert_roundtrip_tree(tree, tmpdir)


def test_equivalency_context_manager():
    base_registry = u.get_current_unit_registry()

    # check starting with only the dimensionless_redshift equivalency.
    assert len(base_registry.equivalencies) == 1
    assert str(base_registry.equivalencies[0][0]) == "redshift"

# ============================================================================
# Descriptor

class CosmologyUnitEquivalenciesTestMixin:
    """
    Test a `astropy.cosmology.units.CosmologyUnitEquivalencies` attached to a |Cosmology|

    This class will not be directly called by :mod:`pytest` since its name does
    not begin with ``Test``. To activate the contained tests this class must
    be inherited in a subclass. Subclasses must define :func:`pytest.fixture`
    ``cosmo`` that returns an instance of a |Cosmology| and ``cosmo_cls`` that
    returns the |Cosmology| class.
    See ``TestCosmology`` for an example.
    """

    @pytest.fixture
    def equivalencies(self, cosmo):
        """``cosmo``-specific equivalencies from descriptor."""
        return cosmo.equivalencies

    # ==============================================================

    def test_equivalencies_available(self, equivalencies):
        """Test ``CosmologyUnitEquivalencies.available``"""
        # check type
        assert isinstance(equivalencies.available, MappingProxyType)
        # check contents
        assert equivalencies.available == equivalencies._equivs
        assert all(map(callable, equivalencies.available.values()))
        #   this test relies on __getattr__, which is also tested
        for n in equivalencies.available.keys():
            equiv = getattr(equivalencies, n)
            assert isinstance(equiv, u.Equivalency)

    def test_equivalencies_dir(self, equivalencies):
        """Test equivalency functions added to ``CosmologyUnitEquivalencies.__dir__``."""
        d = dir(equivalencies)
        assert all([(n in d) for n in equivalencies.available.keys()])


    def test_equivalencies_instance_str(self, cosmo_cls):
        """Test ``CosmologyUnitEquivalencies.__str__``"""
        s = str(cosmo_cls.equivalencies)
        # namelead
        assert repr(cosmo_cls) in s
        # available
        assert all([(n in s) for n in equivalencies.available.keys()])
        # attr dict
        if cosmo_cls.equivalencies._equivs_attrs:  # only check if not empty
            assert all([(v in s) for v in equivalencies._equivs_attrs.values()])

    def test_equivalencies_instance_str(self, cosmo, equivalencies):
        """Test ``CosmologyUnitEquivalencies.__str__``"""
        s = str(equivalencies)
        # namelead
        assert (cosmo.name if cosmo.name is not None else "instance") in s
        assert cosmo.__class__.__qualname__ in s
        # available
        assert all([(n in s) for n in equivalencies.available.keys()])
        # attr dict
        if equivalencies._equivs_attrs:  # only check if not empty
            assert all([(v in s) for v in equivalencies._equivs_attrs.values()]) 

    def test_equivalencies_descriptor(self, cosmo_cls, cosmo):
        """Test accessing equivalencies as a descriptor."""
        # get from class
        equivalencies = cosmo_cls.equivalencies
        assert equivalencies._parent_cls is cosmo_cls
        assert equivalencies._parent_ref is None

        # get from instance
        equivalencies = cosmo.equivalencies 
        assert equivalencies._parent_cls is cosmo_cls
        assert equivalencies._parent_ref is not None
        assert equivalencies._parent_ref() is cosmo

    def test_equivalencies_parent(self, cosmo_cls, cosmo):
        """Test ``CosmologyUnitEquivalencies`` parent."""
        # on the class
        assert cosmo_cls.equivalencies._parent is cosmo_cls

        # on the instance
        assert cosmo.equivalencies._parent is cosmo

    def test_equivalencies_getattr(self, cosmo_cls, equivalencies):
        """Test get equivalencies from ``CosmologyUnitEquivalencies``."""
        # anything from class is forbidden
        # get the first known equivalency
        for attr in tuple(cosmo_cls.equivalencies.available.keys())[:1]:
            with pytest.raises(AttributeError, match=f"equivalency {attr!r}"):
                getattr(cosmo_cls.equivalencies, attr)

        # but is permitted on the instance
        # these equivalencies are instantiated
        for attr in tuple(equivalencies.available.keys())[:1]:
            equiv = getattr(equivalencies, attr)
            assert isinstance(equiv, u.Equivalency)

        # not in equivalency
        with pytest.raises(AttributeError, match=f"equivalency 'not_here'"):
            equivalencies.not_here

    def test_equivalencies_redshift(self, cosmo_cls, cosmo, equivalencies):
        """Test ``CosmologyUnitEquivalencies.redshift``."""
        # errors from the class
        with pytest.raises(AttributeError, match="equivalency 'redshift'"):
            cosmo_cls.equivalencies.redshift

        # works from the instance
        equiv = equivalencies.redshift
        assert isinstance(equiv, u.Equivalency)
        assert equiv.name[0] == "with_redshift"
        assert equiv.kwargs[0]["cosmology"] is cosmo
        assert equiv.kwargs[0]["Tcmb"] == ("redshift_temperature" in equivalencies.available)


# ============================================================================

@pytest.mark.skipif(not HAS_SCIPY, reason = "Cosmology needs scipy")
class Test_unsuitable_units:
    """Ensure that unsuitable units are rejected"""

    def test_scalar(self):
        with pytest.raises(u.UnitConversionError):
            Planck13.Om(0. * u.m)

    @pytest.mark.parametrize('method',
                             ['Om', 'Ob', 'Odm', 'Ode', 'Ogamma', 'Onu', 'Tcmb', 'Tnu',
                              'scale_factor', 'angular_diameter_distance', 'luminosity_distance'])
    def test_unit_exception(self, method):
        with pytest.raises(u.UnitConversionError):
            getattr(Planck13, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method',
                             ['_EdS_age', '_flat_age', '_dS_lookback_time', 'efunc', 'inv_efunc',
                              'Ok'])
    def test_LambdaCDM_unit_exception(self, method):
        LambdaCDMTest = flrw.LambdaCDM(70.0, 0.3, 0.5)
        with pytest.raises(u.UnitConversionError):
            getattr(LambdaCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['efunc', 'inv_efunc'])
    def test_FlatLambdaCDM_unit_exception(self, method):
        FlatLambdaCDMTest = flrw.FlatLambdaCDM(70.0, 0.3)
        with pytest.raises(u.UnitConversionError):
            getattr(FlatLambdaCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['de_density_scale', 'efunc', 'inv_efunc'])
    def test_wCDM_unit_exception(self, method):
        wCDMTest = flrw.wCDM(60.0, 0.27, 0.6, w0 = -0.8)
        with pytest.raises(u.UnitConversionError):
            getattr(wCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['efunc', 'inv_efunc'])
    def test_FlatwCDM_unit_exception(self, method):
        FlatwCDMTest = flrw.FlatwCDM(65.0, 0.27, w0 = -0.6)
        with pytest.raises(u.UnitConversionError):
            getattr(FlatwCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['w', 'de_density_scale'])
    def test_w0waCDM_unit_exception(self, method):
        w0waCDMTest = flrw.w0waCDM(H0 = 70, Om0 = 0.27, Ode0 = 0.5, w0 = -1.2,
                                   wa = -0.2)
        with pytest.raises(u.UnitConversionError):
            getattr(w0waCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['w', 'de_density_scale'])
    def test_wpwaCDM_unit_exception(self, method):
        wpwaCDMTest = flrw.wpwaCDM(H0 = 70, Om0 = 0.27, Ode0 = 0.5, wp = -1.2,
                                   wa = -0.2, zp = 0.9)
        with pytest.raises(u.UnitConversionError):
            getattr(wpwaCDMTest, method)([0., 1.] * u.m)

    @pytest.mark.parametrize('method', ['w', 'de_density_scale'])
    def test_w0wzCDM14_unit_exception(self, method):
        w0wzCDMTest = flrw.w0wzCDM(H0 = 70, Om0 = 0.27, Ode0 = 0.5, w0 = -1.2,
                                   wz = 0.1)
        with pytest.raises(u.UnitConversionError):
            getattr(w0wzCDMTest, method)([0., 1.] * u.m)
