# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""
Read/Write/Interchange methods for `astropy.cosmology`. **NOT public API**.
"""

# Import the interchange to register them into the io registry.
from . import cosmology, ecsv, mapping, model, row, table, yaml  # noqa: F401, F403

try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points


# ==============================================================================
# Entry-Points

def populate_entry_points(entry_points):
    """
    This injects entry points into the `astropy.cosmology.io` namespace.
    This provides a means of inserting a fitting routine without requirement
    of it being merged into astropy's core.

    Parameters
    ----------
    entry_points : list of `~importlib.metadata.EntryPoint`
        entry_points are objects which encapsulate importable objects and
        are defined on the installation of a package.

    Notes
    -----
    An explanation of entry points can be found `here <http://setuptools.readthedocs.io/en/latest/setuptools.html#dynamic-discovery-of-services-and-plugins>`
    """
    from astropy.cosmology.core import Cosmology

    for entry_point in entry_points:
        name = entry_point.name

        # Load entrypoint
        try:
            value = entry_point.load()
        except Exception as e:
            # This stops the fitting from choking if an entry_point produces an error.
            warnings.warn(
                f"{type(e).__name__} error occurred in entry point {name}.",
                AstropyUserWarning)

            return

        # check entrypoint is correct
        allowed_keys = {"read", "write", "to", "from", "identify", "cosmology"}
        if not isinstance(value, dict):
            warnings.warn(
                f"Cosmology IO entry point {name} expected to be a dict.",
                AstropyUserWarning)
            return
        # make sure only have allowed keys
        elif not allowed_keys.issuperset(value.keys()):
            warnings.warn(
                f"entry point {name} can only contain keys {allowed_keys}.",
                AstropyUserWarning)
            return
        # check that if has keys "read/write" can't have keys "to/from"
        elif not (values.keys() - {"to", "from"}).isdisjoint({"read", "write"}):
            warnings.warn(
                f"entry point {name} can only register into read/write or "
                "to/from_format, not both.",
                AstropyUserWarning)
            return
        # check cosmology type
        elif not (inspect.isclass(cosmo_cls := value.get("cosmology", Cosmology))
                  and issubclass(cosmo_cls, Cosmology)):
            warnings.warn(
                f"entry point {name} cosmology must be a Cosmology subclass.",
                AstropyUserWarning)
            return

        # register in any of reader, writer, identifier
        # the name is prepended with the module name, unless they match.
        # the cosmology class defaults to Cosmology
        package_name = str(entry_point.module)
        regname = package_name + name if package_name != name else name
        cosmo_cls = value.get("cosmology", Cosmology)

        if "read" in value:
            reader = value["read"]
            if not callable(reader):
                warnings.warn(
                    f"entry point {name} reader must be a callable.",
                    AstropyUserWarning)
            io_registry.register_reader(regname, cosmo_cls, reader)
        elif "from" in value:
            from_format = value["from"]
            if not callable(from_format):
                warnings.warn(
                    f"entry point {name} `from_format` must be a callable.",
                    AstropyUserWarning)
            io_registry.register_reader(regname, cosmo_cls, from_format)

        if "write" in value:
            writer = value["write"]
            if not callable(writer):
                warnings.warn(
                    f"entry point {name} writer must be a callable.",
                    AstropyUserWarning)
            io_registry.register_reader(regname, cosmo_cls, writer)
        elif "to" in value:
            to_format = value["to"]
            if not callable(to_format):
                warnings.warn(
                    f"entry point {name} `to_format` must be a callable.",
                    AstropyUserWarning)
            io_registry.register_reader(regname, cosmo_cls, to_format)

        if "identify" in value:
            identifier = value["identify"]
            if not callable(writer):
                warnings.warn(
                    f"entry point {name} identifier must be a callable.",
                    AstropyUserWarning)
            io_registry.register_reader(regname, cosmo_cls, identifier)


def _populate_ep():
    # TODO: Exclusively use select when Python minversion is 3.10
    ep = entry_points()
    if hasattr(ep, 'select'):
        populate_entry_points(ep.select(group='astropy_cosmology_io'))
    else:
        populate_entry_points(ep.get('astropy_cosmology_io', []))


_populate_ep()
