# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Testing :mod:`astropy.cosmology.parameter._descriptor`."""

from types import MappingProxyType

import pytest

from astropy.cosmology.parameter import Parameter


class ParameterDescriptorTestMixin:
    """Test the descriptor for ``parameters`` on Cosmology classes."""

    def test_parameters_from_class(self, cosmo_cls):
        """Test descriptor ``parameters`` accessed from the class."""
        # test presence
        assert hasattr(cosmo_cls, "parameters")
        # test Parameter is a MappingProxyType
        assert isinstance(cosmo_cls.parameters, MappingProxyType)
        # Test items
        assert set(cosmo_cls.parameters.keys()) == set(cosmo_cls._parameters_)
        assert all(isinstance(p, Parameter) for p in cosmo_cls.parameters.values())

    def test_parameters_from_instance(self, cosmo):
        """Test descriptor ``parameters`` accessed from the instance."""
        # test presence
        assert hasattr(cosmo, "parameters")
        # test Parameter is a MappingProxyType
        assert isinstance(cosmo.parameters, MappingProxyType)
        # Test keys
        assert set(cosmo.parameters) == set(cosmo._parameters_)

    def test_parameters_cannot_set_on_instance(self, cosmo):
        """Test descriptor ``parameters`` cannot be set on the instance."""
        with pytest.raises(AttributeError, match="cannot set 'parameters' of"):
            cosmo.parameters = {}


class DerivedParameterDescriptorTestMixin:
    """Test the descriptor for ``derived_parameters`` on Cosmology classes."""

    def test_derived_parameters_from_class(self, cosmo_cls):
        """Test descriptor ``derived_parameters`` accessed from the class."""
        # test presence
        assert hasattr(cosmo_cls, "derived_parameters")
        # test Parameter is a MappingProxyType
        assert isinstance(cosmo_cls.derived_parameters, MappingProxyType)
        # Test items
        assert set(cosmo_cls.derived_parameters) == set(cosmo_cls._parameters_derived_)
        assert all(
            isinstance(p, Parameter) for p in cosmo_cls.derived_parameters.values()
        )

    def test_derived_derived_parameters_from_instance(self, cosmo):
        """Test descriptor ``derived_parameters`` accessed from the instance."""
        # test presence
        assert hasattr(cosmo, "derived_parameters")
        # test Parameter is a MappingProxyType
        assert isinstance(cosmo.derived_parameters, MappingProxyType)
        # Test keys
        assert set(cosmo.derived_parameters.keys()) == set(cosmo._parameters_derived_)

    def test_derived_parameters_cannot_set_on_instance(self, cosmo):
        """Test descriptor ``derived_parameters`` cannot be set on the instance."""
        with pytest.raises(AttributeError):
            cosmo.derived_parameters = {}
