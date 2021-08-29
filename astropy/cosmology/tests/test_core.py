# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Testing :mod:`astropy.cosmology.core`."""

##############################################################################
# IMPORTS

import abc
import inspect
import os
import pickle
from io import StringIO

import pytest

import numpy as np

import astropy.constants as const
import astropy.io.registry as io_registry
import astropy.units as u
from astropy.cosmology import Cosmology, CosmologyError
from astropy.cosmology.connect import (CosmologyFromFormat, CosmologyRead,
                                       CosmologyToFormat, CosmologyWrite)
from astropy.tests.helper import check_pickling_recovery, pickle_protocol
from astropy.units import allclose
from astropy.utils.compat.optional_deps import HAS_SCIPY

##############################################################################
# TESTS
##############################################################################


def test_CosmologyError():
    """Test :class:`astropy.cosmology.CosmologyError`."""
    # a most basic test
    with pytest.raises(CosmologyError):
        raise CosmologyError

    # error from usage
    # TODO!


# ----------------------------------------------------------------------------


class TestCosmology:
    """Test :class:`astropy.cosmology.Cosmology`"""

    def setup_class(self):
        """Setup class for pytest."""
        self.cls = Cosmology
        self.cls_args = (70 * (u.km / u.s / u.Mpc),)  # nothing real
        self.cls_kwargs = dict(name="test", meta={"a": "b"})

    @pytest.fixture(scope="class")
    def cosmo_class(self):
        return self.cls

    @pytest.fixture
    def cosmo(self, request):
        """The cosmology instance with which to test."""
        return self.cls(*self.cls_args, **self.cls_kwargs)

    @pytest.fixture(params=["units", "dmnls"])
    def cosmos(self, request):
        """
        The cosmology instance with which to test.
        Will be repeated: one with units, once without.
        """
        if request.param == "units":
            args = self.cls_args
            kwargs = self.cls_kwargs
        else:  # strip units
            args = (u.Quantity(x).value for x in self.cls_args)
            kwargs = {
                k: (v.to_value() if isinstance(v, u.Quantity) else v)
                for k, v in self.cls_kwargs.items()
            }

        return self.cls(*args, **kwargs)

    # ===============================================================
    # Method & Attribute Tests

    def test_init_subclass(self, cosmo_class, clean_cosmo_registry):
        """Test ``Cosmology.__init_subclass__``."""

        class SubClass(cosmo_class):
            pass

        assert SubClass in clean_cosmo_registry.values()
        assert f"TestCosmology.test_init_subclass.<locals>.SubClass" in clean_cosmo_registry.keys()

    # -------------------------------------------

    def test_new(self, cosmo_class):
        """Test :meth:`astropy.cosmology.Cosmology.__new__`."""
        # make self without calling __init__
        cosmo = cosmo_class.__new__(cosmo_class, *self.cls_args, **self.cls_kwargs)

        # test name and meta not set / empty (from __init__)
        assert not hasattr(cosmo, "_name")
        assert not cosmo.meta

        # test stores initialization arguments on the instance
        ba = cosmo_class._init_signature.bind_partial(*self.cls_args, **self.cls_kwargs)
        ba.apply_defaults()
        assert cosmo._init_arguments == ba.arguments

    # -------------------------------------------

    def _cosmo_test_init_attr(self, cosmo):
        """Helper function for testing ``__init__``"""
        assert hasattr(cosmo, "_name")
        assert cosmo._name is None or isinstance(cosmo._name, str)

        assert hasattr(cosmo, "meta")
        assert isinstance(cosmo.meta, dict)

    def test_init(self, cosmo_class):
        # Cosmology accepts any args, kwargs
        cosmo1 = cosmo_class(1, 2, 3, 4, 5, a=1, b=2, c=3, d=4, e=5)
        self._cosmo_test_init_attr(cosmo1)

        # but only "name" and "meta" are used
        cosmo2 = cosmo_class(name="test", meta={"m": 1})
        self._cosmo_test_init_attr(cosmo2)
        assert cosmo2.name == "test"
        assert cosmo2.meta["m"] == 1

    # -------------------------------------------
    # Properties

    def test_init_signature(self, cosmos):
        # hopefully
        assert isinstance(cosmos._init_signature, inspect.Signature)
        # all parameters except self
        sig = inspect.signature(cosmos.__class__.__init__)
        assert "self" in sig.parameters
        assert "self" not in cosmos._init_signature.parameters
        # test equal
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])
        assert sig.parameters == cosmos._init_signature.parameters

    @pytest.mark.skip("TODO!")
    def test_meta(self, cosmo):
        assert False

    def test_name(self, cosmo):
        # test property
        assert cosmo.name is cosmo._name

        # expected value
        assert cosmo.name == self.cls_kwargs.get("name")

        # and just to make sure (if name is None)
        kw = self.cls_kwargs | {"name": "test"}
        cosmo2 = self.cls(*self.cls_args, **kw)
        assert cosmo2.name == "test"

        # test immutability
        with pytest.raises(AttributeError):
            cosmo.name = "new name"

    # -------------------------------------------

    def test_clone(self, cosmo):
        """Test clone operation."""
        # First, test with no changes, which should return same object
        newclone = cosmo.clone()
        assert newclone is cosmo

        # Now change 1st argument
        k, v = list(cosmo._init_arguments.items())[0]  # get 1st argument
        newclone = cosmo.clone(**{k: v * 2})
        assert newclone is not cosmo
        assert newclone.__class__ == cosmo.__class__
        assert newclone.name == None if cosmo.name is None else cosmo.name + " (modified)"

    def test_clone_fail_positional_arg(self, cosmo):
        """Test clone fails if pass positional argument."""
        with pytest.raises(TypeError, match="1 positional argument"):
            cosmo.clone(None)

    # -------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_equality(self, cosmos):
        """Test ``__eq__``."""
        assert False

    # ===============================================================
    # I/O
    # one reason these tests are here, and not in ``test_connect`` is so that
    # they are called in each Cosmology's test class and therefore test every
    # type of cosmology.

    def test_read_property(self, cosmo_class):
        """Test ``read``."""
        assert isinstance(cosmo_class.read, CosmologyRead)

    def test_write_property(self, cosmos):
        """Test ``write``."""
        assert isinstance(cosmos.write, CosmologyWrite)

    def test_write_methods_have_explicit_kwarg_overwrite(self, readwrite_format):
        writer = io_registry.get_writer(readwrite_format, Cosmology)
        # test in signature
        sig = inspect.signature(writer)
        assert "overwrite" in sig.parameters

        # also in docstring
        assert "overwrite : bool" in writer.__doc__

    def test_readwrite_complete_info(self, cosmos, tmpdir, readwrite_format):
        """
        Test writing from an instance and reading from the base class.
        This requires full information.
        """
        fname = tmpdir / f"{cosmos.name}.{readwrite_format}"

        cosmos.write(str(fname), format=readwrite_format)

        # Also test kwarg "overwrite"
        assert os.path.exists(str(fname))  # file exists
        with pytest.raises(IOError):
            cosmos.write(str(fname), format=readwrite_format, overwrite=False)

        assert os.path.exists(str(fname))  # overwrite file existing file
        cosmos.write(str(fname), format=readwrite_format, overwrite=True)

        # Read back
        got = Cosmology.read(fname, format=readwrite_format)

        assert got == cosmos
        assert got.meta == cosmos.meta

    def test_readwrite_from_subclass_complete_info(self, cosmos, tmpdir, readwrite_format):
        """
        Test writing from an instance and reading from that class, when there's
        full information saved.
        """
        fname = tmpdir / f"{cosmos.name}.{readwrite_format}"
        cosmos.write(str(fname), format=readwrite_format)

        # read with the same class that wrote.
        got = cosmos.__class__.read(fname, format=readwrite_format)
        assert got == cosmos
        assert got.meta == cosmos.meta

        # this should be equivalent to
        got = Cosmology.read(fname, format=readwrite_format, cosmology=cosmos.__class__)
        assert got == cosmos
        assert got.meta == cosmos.meta

        # and also
        got = Cosmology.read(fname, format=readwrite_format,
                             cosmology=cosmos.__class__.__qualname__)
        assert got == cosmos
        assert got.meta == cosmos.meta

    @pytest.mark.skip("TODO!")
    def test_readwrite_from_subclass_partial_info(self, cosmos, tmpdir, readwrite_format):
        """
        Test writing from an instance and reading from that class.
        This requires partial information.

        .. todo::

            generalize over all save formats for this test.
        """
        assert False

    @pytest.mark.skip("TODO!")
    def test_reader_class_mismatch(self, cosmos, tmpdir, readwrite_format):
        """Test when the reader class doesn't match the file."""
        assert False

    # -------------------------------------------

    def test_from_format_property(self, cosmo_class):
        """Test ``from_format``."""
        assert isinstance(cosmo_class.from_format, CosmologyFromFormat)

    def test_to_format_property(self, cosmos):
        """Test ``to_format``."""
        assert isinstance(cosmos.to_format, CosmologyToFormat)

    def test_format_complete_info(self, cosmos, tofrom_format):
        """Read tests happen later."""
        format, objtype = tofrom_format

        # test to_format
        obj = cosmos.to_format(format)
        assert isinstance(obj, objtype)

        # test from_format
        got = Cosmology.from_format(obj, format=format)
        # and autodetect
        got2 = Cosmology.from_format(obj)

        assert got2 == got  # internal consistency
        assert got == cosmos  # external consistency
        assert got.meta == cosmos.meta

    def test_format_from_subclass_complete_info(self, cosmos, tofrom_format):
        """
        Test transforming an instance and parsing from that class, when there's
        full information available.
        """
        format, objtype = tofrom_format

        # test to_format
        obj = cosmos.to_format(format)
        assert isinstance(obj, objtype)

        # read with the same class that wrote.
        got = cosmos.__class__.from_format(obj, format=format)
        got2 = Cosmology.from_format(obj)  # and autodetect

        assert got2 == got  # internal consistency
        assert got == cosmos  # external consistency
        assert got.meta == cosmos.meta

        # this should be equivalent to
        got = Cosmology.from_format(obj, format=format, cosmology=cosmos.__class__)
        assert got == cosmos
        assert got.meta == cosmos.meta

        # and also
        got = Cosmology.from_format(obj, format=format, cosmology=cosmos.__class__.__qualname__)
        assert got == cosmos
        assert got.meta == cosmos.meta

    @pytest.mark.skip("TODO!")
    def test_format_from_subclass_partial_info(self, cosmos):
        """
        Test writing from an instance and reading from that class.
        This requires partial information.

        .. todo::

            generalize over all formats for this test.
        """
        assert False

    @pytest.mark.skip("TODO!")
    def test_format_reader_class_mismatch(self, cosmos, tofrom_format):
        """Test when the reader class doesn't match the object."""
        assert False

    # ===============================================================
    # Usage Tests

    def test_immutability(self, cosmo):
        """Test immutability of cosmologies."""
        with pytest.raises(AttributeError):
            cosmo.name = "new name"
        # TODO! more stringent test of general immutability

        # The metadata is NOT immutable
        assert "mutable" not in cosmo.meta
        cosmo.meta["mutable"] = True
        assert "mutable" in cosmo.meta
        cosmo.meta.pop("mutable")
        assert "mutable" not in cosmo.meta

    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4])  # add [5] when drop 3.7
    def test_pickle_class(self, cosmo_class, protocol):
        """Test cosmology classes can be pickled"""
        # skip this test for "local" objects, like in the FLRW test
        if "<locals>" in cosmo_class.__qualname__:
            pytest.xfail()
        check_pickling_recovery(cosmo_class, protocol)

    @pytest.mark.parametrize("protocol", [0, 1, 2, 3, 4])  # add [5] when drop 3.7
    def test_pickle_realization(self, cosmo, protocol):
        """
        Test realizations can pickle and unpickle.
        Also a regression test for #12008.
        """
        # skip this test for "local" objects, like in the FLRW test
        if "<locals>" in cosmo.__class__.__qualname__:
            pytest.xfail()

        # pickle and unpickle
        f = pickle.dumps(cosmo, protocol=protocol)
        unpickled = pickle.loads(f)
    
        # test equality
        assert unpickled == cosmo
        assert unpickled.meta == cosmo.meta
