# -*- coding: utf-8 -*-

"""Decorators for configuration."""


__all__ = [
    "replace_from_config",
]


##############################################################################
# IMPORTS

# BUILT-IN
import functools
import inspect

##############################################################################
# CODE
##############################################################################


def replace_from_config(
    function=None, *, config, replace=Ellipsis,
):
    """Replace config inputs to functions with their config values.

    Parameters
    ----------
    function : callable or None, optional
        the function to be decoratored
        if None, then returns decorator to apply.
    config : :class:`~astropy.config.ConfigNamespace` instance
        The configuration instance.
    replace : list of strings or Ellipsis (optional, key-word only)
        Function arguments to be mapped from None to their `config` value.
        The name of the input must match a `config` attribute.
        If Ellipsis (default), tries to match all names in `config`

    Returns
    -------
    wrapper : callable
        wrapper for function

    Examples
    --------
    >>> from astropy.utils.data import conf
    >>> @replace_from_config(config=conf)
    ... def return_dataurl(dataurl=None):
    ...     return dataurl
    >>> return_dataurl()
    'http://data.astropy.org/'

    """
    if function is None:  # allowing for optional arguments
        return functools.partial(
            replace_from_config, config=config, replace=replace,
        )

    if replace is Ellipsis:
        replace = config.keys()  # keep as view so updates
    elif isinstance(replace, str):  # type munging
        replace = [replace]

    sig = inspect.signature(function)

    @functools.wraps(function)
    def wrapper(*args, **kw):
        """Wrapper docstring."""
        ba = sig.bind_partial(*args, **kw)
        ba.apply_defaults()  # need this to check all arguments

        for name in replace:
            # check if should replace from config
            # skips if not in func sig
            if ba.arguments.get(name, False) is None:
                ba.arguments[name] = getattr(config, name)

        return function(*ba.args, **ba.kwargs)

    return wrapper
