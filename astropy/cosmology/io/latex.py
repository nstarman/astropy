import astropy.units as u
from astropy.cosmology.connect import readwrite_registry
from astropy.cosmology.core import Cosmology
from astropy.cosmology.parameter import Parameter
from astropy.table import QTable


from .table import to_table
from .utils import _FORMAT_TABLE


def write_latex(
    cosmology, file, *, overwrite=False, cls=QTable, latex_names=True, **kwargs
):
    r"""Serialize the |Cosmology| into a LaTeX.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology` subclass instance
    file : path-like or file-like
        Location to save the serialized cosmology.

    overwrite : bool
        Whether to overwrite the file, if it exists.
    cls : type, optional keyword-only
        Astropy :class:`~astropy.table.Table` (sub)class to use when writing.
        Default is :class:`~astropy.table.QTable`.
    latex_names : bool, optional keyword-only
        Whether to use LaTeX names for the parameters. Default is `True`.
    **kwargs
        Passed to ``cls.write``

    Raises
    ------
    TypeError
        If kwarg (optional) 'cls' is not a subclass of `astropy.table.Table`
    """
    # Check that the format is 'latex' (or not specified)
    format = kwargs.pop("format", "latex")
    if format != "latex":
        raise ValueError(f"format must be 'latex', not {format}")

    # Set cosmology_in_meta as false for now since there is no metadata being kept
    table = to_table(cosmology, cls=cls, cosmology_in_meta=False)

    cosmo_cls = type(cosmology)
    for name, col in table.columns.copy().items():
        param = getattr(cosmo_cls, name, None)
        if not isinstance(param, Parameter) or param.unit in (None, u.one):
            continue
        # Get column to correct unit
        table[name] <<= param.unit

    # Convert parameter names to LaTeX format
    if latex_names:
        new_names = [_FORMAT_TABLE.get(k, k) for k in cosmology.__parameters__]
        table.rename_columns(cosmology.__parameters__, new_names)

    table.write(file, overwrite=overwrite, format="latex", **kwargs)


# ===================================================================
# Register

readwrite_registry.register_writer("latex", Cosmology, write_latex)
