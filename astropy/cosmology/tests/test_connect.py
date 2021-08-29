# Licensed under a 3-clause BSD style license - see LICENSE.rst

import copy
import inspect
import json
import os

import pytest

import numpy as np

import astropy.cosmology
import astropy.units as u
from astropy.cosmology import Cosmology, Planck18
from astropy.cosmology.connect import (CosmologyFromFormat, CosmologyRead,
                                       CosmologyToFormat, CosmologyWrite)
from astropy.cosmology.parameters import available

###############################################################################
# Prepare

class CosmologyIOClassTest:
    
    @pytest.fixture(params=available)
    def cosmo(self, request):
        """Cosmology instances"""
        return getattr(astropy.cosmology, request.param)

    @pytest.fixture
    def ioclass(self):
        """The I/O class, e.g. ``CosmologyRead``."""
        return self.cls

    @pytest.fixture
    def instance(self, ioclass, cosmo):
        """
        An instance of the I/O class, e.g. ``CosmologyRead``.
        """
        return ioclass(cosmo, cosmo.__class__)

    # ===============================================================

    def test_init(self, ioclass, cosmo):
        """Test ``__init__``."""
        instance = ioclass(cosmo, cosmo.__class__)

        assert isinstance(instance, ioclass)
        assert instance._method_name == self.method_name

    @pytest.mark.skip("TODO!")
    def test_help(self):
        assert False

    @pytest.mark.skip("TODO!")
    def list_formats(self):
        assert False


###############################################################################
# Tests


class TestCosmologyRead(CosmologyIOClassTest):
    """
    Test :class:`astropy.cosmology.connect.CosmologyRead`.
    Most of the real tests are on ``test_core.TestCosmology`` and subclasses,
    since :class:`astropy.cosmology.Cosmology` has the actual read method.
    """

    def setup_class(self):
        self.cls = CosmologyRead
        self.method_name = "read"

    # ===============================================================

    def test_call_wrong_cosmology(self, instance):
        """Test ``__call__`` when it is the wrong Cosmology type."""
        with pytest.raises(ValueError, match="keyword argument `cosmology`"):
            instance(cosmology=Cosmology)

    def test_call(self):
        pass  # this is tested in ``test_core``
    


class TestCosmologyWrite:
    """
    Test :class:`astropy.cosmology.connect.CosmologyWrite`.
    Most of the real tests are on ``test_core.TestCosmology`` and subclasses,
    since :class:`astropy.cosmology.Cosmology` has the actual write method.
    """

    def setup_class(self):
        self.cls = CosmologyRead
        self.method_name = "write"

    # ===============================================================

    def test_call(self):
        pass  # this is tested in ``test_core``


@pytest.mark.skip("TODO!")
class TestCosmologyFromFormat:
    """
    Test :class:`astropy.cosmology.connect.CosmologyFromFormat`.
    Most of the real tests are on ``test_core.TestCosmology`` and subclasses,
    since :class:`astropy.cosmology.Cosmology` has the actual format method.
    """

    def setup_class(self):
        self.cls = CosmologyRead
        self.method_name = "read"

    # ===============================================================

    def test_call_wrong_cosmology(self, instance):
        """Test ``__call__`` when it is the wrong Cosmology type."""
        with pytest.raises(ValueError, match="keyword argument `cosmology`"):
            instance(cosmology=Cosmology)

    def test_call(self):
        pass  # this is tested in ``test_core``


class TestCosmologyToFormat:
    """
    Test :class:`astropy.cosmology.connect.CosmologyToFormat`.
    Most of the real tests are on ``test_core.TestCosmology`` and subclasses,
    since :class:`astropy.cosmology.Cosmology` has the actual format method.
    """

    def setup_class(self):
        self.cls = CosmologyRead
        self.method_name = "write"

    # ===============================================================

    def test_call(self):
        pass  # this is tested in ``test_core``
