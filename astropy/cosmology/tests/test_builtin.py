# -*- coding: utf-8 -*-

"""Test :mod:`~astropy.cosmology.builtin`."""

from types import MappingProxyType

import pytest

from .. import builtin, core


def test_parameter_registry():
    """
    Test `~astropy.cosmology.builtin._parameter_registry`.

    """
    registry = builtin._parameter_registry

    assert isinstance(registry, dict)

    minkeys = {"parameters", "references", "cosmo"}

    for n, state in registry.items():
        assert isinstance(state, (MappingProxyType, dict))
        assert minkeys.issubset(state.keys())

        assert n in builtin.available


def test_default_cosmology():
    """
    Test `~astropy.cosmology.builtin.default_cosmology`.

    """
    # Attributes test

    assert builtin.default_cosmology._default_value == "Planck15"
    assert builtin.default_cosmology._registry == builtin._parameter_registry
    assert builtin.default_cosmology.available == builtin.available

    # ----------------------------------------
    # Get and Validate

    cosmo = builtin.default_cosmology.validate(None)
    assert cosmo == builtin.Planck15

    cosmo = builtin.default_cosmology.get_cosmology_from_string("no_default")
    assert cosmo is None

    # not in registry
    with pytest.raises(ValueError):
        cosmo = builtin.default_cosmology.get_cosmology_from_string("test")

    # ----------------------------------------
    # Registration tests

    test_parameters = dict(
        Oc0=0.231,
        Ob0=0.0459,
        Om0=0.277,
        Ode0=0.4461,
        H0=70.2,
        n=0.962,
        sigma8=0.817,
        tau=0.088,
        z_reion=11.3,
        t0=13.72,
        Tcmb0=2.725,
        Neff=3.04,
        m_nu=0.0,
        flat=False,  # Needed to hit LambdaCDM, not only FlatLambdaCDM
    )

    try:  # register_parameters
        builtin.default_cosmology.register_parameters(
            name="test",
            parameters=test_parameters,
            references=None,
            cosmo=None,
            viewonly=True,
        )
        assert "test" in builtin._parameter_registry

        # test get_from_registry
        state = builtin.default_cosmology.get_from_registry("test")

        assert state["parameters"] == test_parameters
        assert state["references"] is None
        assert state["cosmo"] is None

        with builtin.default_cosmology.set("test"):
            test_cosmo = builtin.default_cosmology.get()

    finally:
        builtin._parameter_registry.pop("test", None)
        assert "test" not in builtin._parameter_registry

    # ----------------------------------------
    # Try failures
    test_parameters["flat"] = False
    test_parameters.pop("Ode0")

    try:  # register_parameters
        with pytest.raises(ValueError):
            builtin.default_cosmology.register_parameters(
                name="test", parameters=test_parameters
            )

    finally:
        builtin._parameter_registry.pop("test", None)
        assert "test" not in builtin._parameter_registry

    # ----------------------------------------

    try:  # register_cosmology_instance
        builtin.default_cosmology.register_cosmology_instance(cosmo=test_cosmo)

        assert "test" in builtin._parameter_registry
        assert builtin._parameter_registry["test"]["references"] is None
        assert builtin._parameter_registry["test"]["cosmo"] == core.LambdaCDM

        with builtin.default_cosmology.set("test"):
            test2_cosmo = builtin.default_cosmology.get()

        assert test2_cosmo == test_cosmo

    finally:
        builtin._parameter_registry.pop("test", None)
        assert "test" not in builtin._parameter_registry
