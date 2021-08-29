# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Configure the tests for :mod:`astropy.cosmology`."""

##############################################################################
# IMPORTS

import json
import os

import pytest

import astropy.units as u
from astropy.cosmology import Cosmology, core
from astropy.io import registry as io_registry

###############################################################################
# Setup


def read_json(filename, **kwargs):
    with open(filename, "r") as file:
        data = file.read()
    mapping = json.loads(data)  # parse json mappable to dict
    # deserialize Quantity
    for k, v in mapping.items():
        if isinstance(v, dict) and "value" in v and "unit" in v:
            mapping[k] = u.Quantity(v["value"], v["unit"])
        elif isinstance(v, (tuple, list)):
            vv = [(w if not (isinstance(w, dict) and "value" in w and "unit" in w)
                   else u.Quantity(w["value"], w["unit"]))
                  for w in v]
            mapping[k] = tuple(vv)
    for k, v in mapping.get("meta", {}).items():  # also the metadata
        if isinstance(v, dict) and "value" in v and "unit" in v:
            mapping["meta"][k] = u.Quantity(v["value"], v["unit"])
        elif isinstance(v, tuple):
            vv = [(w if not (isinstance(w, dict) and "value" in w and "unit" in w)
                   else u.Quantity(w["value"], w["unit"]))
                  for w in v]
            mapping["meta"][k] = tuple(vv)
    return Cosmology.from_format(mapping, **kwargs)


def write_json(cosmology, file, *, overwrite=False):
    """Write Cosmology to JSON.

    Parameters
    ----------
    cosmology : `astropy.cosmology.Cosmology` subclass instance
    file : path-like or file-like
    overwrite : bool (optional, keyword-only)
    """
    data = cosmology.to_format("mapping")  # start by turning into dict
    data["cosmology"] = data["cosmology"].__qualname__
    # serialize Quantity
    for k, v in data.items():
        if isinstance(v, u.Quantity):
            data[k] = {"value": v.value.tolist(), "unit": str(v.unit)}
        elif isinstance(v, tuple):
            vv = [(w if not isinstance(w, u.Quantity)
                   else {"value": w.value.tolist(), "unit": str(w.unit)})
                  for w in v]
            data[k] = tuple(vv)
    for k, v in data.get("meta", {}).items():  # also serialize the metadata
        if isinstance(v, u.Quantity):
            data["meta"][k] = {"value": v.value.tolist(), "unit": str(v.unit)}
        elif isinstance(v, tuple):
            vv = [(w if not isinstance(w, u.Quantity)
                   else {"value": w.value.tolist(), "unit": str(w.unit)})
                  for w in v]
            data["meta"][k] = tuple(vv)

    # check that file exists and whether to overwrite.
    if os.path.exists(file) and not overwrite:
        raise IOError(f"{file} exists. Set 'overwrite' to write over.")
    with open(file, "w") as write_file:
        json.dump(data, write_file)


def json_identify(origin, filepath, fileobj, *args, **kwargs):
    return filepath is not None and filepath.endswith(".json")


##############################################################################


@pytest.fixture
def clean_cosmo_registry():
    # TODO! with monkeypatch instead for thread safety.
    ORIGINAL_COSMOLOGY_CLASSES = core._COSMOLOGY_CLASSES
    core._COSMOLOGY_CLASSES = {}  # set as empty dict

    yield core._COSMOLOGY_CLASSES

    core._COSMOLOGY_CLASSES = ORIGINAL_COSMOLOGY_CLASSES


@pytest.fixture(params=["json"])
def readwrite_format(request):
    """Setup and teardown read/write in Cosmology I/O for a test."""
    format = request.param
    if format == "json":
        io_registry.register_reader(format, Cosmology, read_json)
        io_registry.register_writer(format, Cosmology, write_json)
        io_registry.register_identifier(format, Cosmology, json_identify)
    else:
        raise Exception

    yield format  # now call test function

    io_registry.unregister_reader(format, Cosmology)
    io_registry.unregister_writer(format, Cosmology)
    io_registry.unregister_identifier(format, Cosmology)


@pytest.fixture(
    params=(set(io_registry.get_formats(Cosmology, "Read")["Format"])
            & (set(io_registry.get_formats(Cosmology, "Write")["Format"]))))
def tofrom_format(request):
    """Setup and teardown JSON in Cosmology I/O for a test."""
    format = request.param
    if format == "mapping":
        cls = dict
    else:
        raise Exception

    yield (format, cls)  # now call test function
