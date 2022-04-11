# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
This module provides the tools used to internally run the cosmology test suite
from the installed astropy.  It makes use of the `pytest`_ testing framework.
"""

##############################################################################
# IMPORTS

# STDLIB
from importlib.resources import path
import inspect
import pathlib

# THIRD PARTY
import numpy as np
import pytest

# LOCAL
import astropy.cosmology.units as cu
import astropy.units as u
from astropy.cosmology import core
from astropy.io.misc.yaml import dump, load
from astropy.utils.introspection import find_current_module

# from astropy.table import Column

__all__ = ["get_redshift_methods", "clean_registry"]

###############################################################################
# PARAMETERS

scalar_zs = [
    0, 1, 1100,  # interesting times
    # FIXME! np.inf breaks some funcs. 0 * inf is an error
    np.float64(3300),  # different type
    2 * cu.redshift, 3 * u.one  # compatible units
]
_zarr = np.linspace(0, 1e5, num=20)
array_zs = [
    _zarr,  # numpy
    _zarr.tolist(),  # pure python
    # Column(_zarr),  # table-like  # TODO!
    _zarr * cu.redshift  # Quantity
]
valid_zs = scalar_zs + array_zs

invalid_zs = [
    (None, TypeError),  # wrong type
    # Wrong units (the TypeError is for the cython, which can differ)
    (4 * u.MeV, (u.UnitConversionError, TypeError)),  # scalar
    ([0, 1] * u.m, (u.UnitConversionError, TypeError)),  # array
]


###############################################################################
# FUNCTIONS


def get_redshift_methods(cosmology, include_private=True, include_z2=True):
    """Get redshift methods from a cosmology.

    Parameters
    ----------
    cosmology : |Cosmology| class or instance
    include_private : bool
        Whether to include private methods, i.e. starts with an underscore.
    include_z2 : bool
        Whether to include methods that are functions of 2 (or more) redshifts,
        not the more common 1 redshift argument.

    Returns
    -------
    set[str]
        The names of the redshift methods on `cosmology`, satisfying
        `include_private` and `include_z2`.
    """
    # Get all the method names, optionally sieving out private methods
    methods = set()
    for n in dir(cosmology):
        try:  # get method, some will error on ABCs
            m = getattr(cosmology, n)
        except NotImplementedError:
            continue

        # Add anything callable, optionally excluding private methods.
        if callable(m) and (not n.startswith('_') or include_private):
            methods.add(n)

    # Sieve out incompatible methods.
    # The index to check for redshift depends on whether cosmology is a class
    # or instance and does/doesn't include 'self'.
    iz1 = 1 if inspect.isclass(cosmology) else 0
    for n in tuple(methods):
        try:
            sig = inspect.signature(getattr(cosmology, n))
        except ValueError:  # Remove non-introspectable methods.
            methods.discard(n)
            continue
        else:
            params = list(sig.parameters.keys())

        # Remove non redshift methods:
        if len(params) <= iz1:  # Check there are enough arguments.
            methods.discard(n)
        elif len(params) >= iz1 + 1 and not params[iz1].startswith("z"):  # First non-self arg is z.
            methods.discard(n)
        # If methods with 2 z args are not allowed, the following arg is checked.
        elif not include_z2 and (len(params) >= iz1 + 2) and params[iz1 + 1].startswith("z"):
            methods.discard(n)

    return methods


# ============================================================================
# Test Generation


def generate_all_tests_results_files():

    # collect all the tests
    from astropy.cosmology.flrw.tests import (test_base, test_lambdacdm,  # noqa: F401, F403
                                              test_w0cdm, test_w0wacdm, test_w0wzcdm,
                                              test_wpwazpcdm)
    from astropy.cosmology.tests.test_core import TestCosmology

    def _recursively_generate_test_result(test_class):
        # generate results for this class
        if not getattr(test_class, "__abstractmethods__", ()):
            print(test_class)

            _, results = test_class.generate_test_results(valid_zs)

            DATA_DIR = pathlib.Path(inspect.getfile(test_class)).parent / "data"
            DATA_DIR.mkdir(exist_ok=True)
            path = DATA_DIR / f"data_{test_class.__name__}.yml"

            with open(path, "w") as f:
                dump(dict(results), f)

        # recurse through all subclasses
        for cls in test_class.__subclasses__():
            _recursively_generate_test_result(cls)

    _recursively_generate_test_result(TestCosmology)


def load_test_results(test_class_name):
    module = find_current_module(depth=2)
    DATA_DIR = pathlib.Path(module.__file__).parent / "data"
    path = DATA_DIR / f"data_{test_class_name}.yml"

    with open(path, "r") as f:
        with u.add_enabled_units(cu):
            result = load(f)

    return result


###############################################################################
# FIXTURES

@pytest.fixture
def clean_registry():
    """`pytest.fixture` for clearing and restoring ``_COSMOLOGY_CLASSES``."""
    # TODO! with monkeypatch instead for thread safety.
    ORIGINAL_COSMOLOGY_CLASSES = core._COSMOLOGY_CLASSES
    core._COSMOLOGY_CLASSES = {}  # set as empty dict

    yield core._COSMOLOGY_CLASSES

    core._COSMOLOGY_CLASSES = ORIGINAL_COSMOLOGY_CLASSES
