# Licensed under a 3-clause BSD style license - see LICENSE.rst

import functools
import inspect
import weakref
from collections import defaultdict
from types import FunctionType

from astropy.utils.collections import ClassWrapperMeta
from astropy.utils.compat.optional_deps import HAS_SCIPY

from .core import Cosmology

# isort: split
if HAS_SCIPY:
    from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline
else:

    class InterpolatedUnivariateSpline:
        def __call__(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'scipy.interpolate'")

    class RectBivariateSpline:
        def __call__(*args, **kwargs):
            raise ModuleNotFoundError("No module named 'scipy.interpolate'")


__all__ = ["Interpolated", "InterpolatedCosmology"]


class InterpolatedMeta(ClassWrapperMeta):

    _interp_func_info = defaultdict(dict)
    """Registry of information how to interpolate a method on a class.
    ``_interp_func_info[Class.__qualname__][method.__name__][argname]=dict(info)``
    """

    def __call__(cls, *args, **kwargs):
        # TODO! integrate with super
        if inspect.isclass(args[0]):
            if issubclass(args[0], cls.mro()[1]):
                return cls._get_wrapped_subclass(args[0])
            else:
                raise TypeError(f"must be a subclass of {cls}, not {args[0]}.")

        return super().__call__(*args, **kwargs)

    def _make_wrapper_subclass(cls, data_cls, base_cls):
        return type("Interpolated" + data_cls.__name__,
                    (data_cls, base_cls), {}, data_cls=data_cls)

    def _get_wrapper_subclass_instance(cls, data):
        raise NotImplementedError("TODO!")


class Interpolated(metaclass=InterpolatedMeta,
                   default_wrapped_class="InterpolatedCosmology"):
    """A Cosmology with its methods interpolated over a redshift range.

    The resulting instance will take its exact type from whatever the
    contents are, with the type generated on the fly as needed.

    Parameters
    ----------
    *args, **kwargs
        - If creating from a subclass, just super().
        - for factory mode: no kwargs, only 1 arg. |Cosmology| class or
          instance thereof.

    Raises
    ------
    ValueError
        If ``cls`` is Interpolated and the arguments are not for "factory" mode
        -- there cannot be any kwargs and only one arg.
    TypeError
        If in "factory" mode and the only arg is not a |Cosmology| class or
        instance thereof.
    """

    def __init_subclass__(cls, data_class=None, base_class=None, **kwargs):
        """Register an Interpolated subclass.

        Parameters
        ----------
        **kwargs
            Passed on for possible further initialization by superclasses.
        """
        # Make interpolation wrapper versions of methods
        # work through the base classes in order of MRO looking for interpolation
        # info set in ``@register``. The MRO tells us which info to use when
        # there are multiple from different classes.
        already_wrapped = set()
        for i, mro_item in enumerate(cls.__mro__):
            # skip this MRO if nothing registered for this class
            if mro_item.__qualname__ not in Interpolated._interp_func_info:
                continue

            # get info on interpolated method
            imethinfo = Interpolated._interp_func_info[mro_item.__qualname__]

            # have the class, now need to work through all the methods
            for n, ikw in imethinfo.items():
                # if method somehow removed or no longer callable, skip it
                if n not in dir(cls) or not callable(getattr(cls, n)):
                    continue
                # find superseded methods that are no longer interpolated
                # E.g. for `class A: def func(self): pass` and `class B(A):`
                # then `B.func` is <function A.func(self)>. If B overwrites
                # `func`, then it's <function B.func(self)> .
                if mro_item.__qualname__ not in getattr(cls, n).__qualname__:
                    already_wrapped.add(n)
                    continue

                # else, make interpolation wrapper for method
                cls._make_interpolation_wrapper(n, **ikw)
                # and register it so don't rewrap
                already_wrapped.add(n)

        # keep going with subclass initialization, e.g. in Cosmology
        kwargs.pop("data_cls", None)
        kwargs.pop("base_cls", None)
        super().__init_subclass__(**kwargs)

    @classmethod
    def _make_interpolation_wrapper(cls, method_name, **interpkw):
        """Make interpolation wrapper for method.

        .. todo::

            - have a flag for pre-computed when initialize class,
              currently this is JIT

        Parameters
        ----------
        method_name : str
            The name of the method on this class to be interpolated and wrapped.
        **interpkw
            Keyword arguments for the interpolation function.
            - `~scipy.interpolate.InterpolatedUnivariateSpline` or
            - `~scipy.interpolate.RectBivariateSpline`
        """
        # get method from class. This is an unbound method, so will require
        # passing "self" when evaluated.
        method = getattr(cls, method_name)

        # get signature, dropping "self" by taking arguments [1:]
        sig = inspect.signature(method)
        sig = sig.replace(parameters=list(sig.parameters.values())[1:])

        # find the positional parameters
        pos_params = {k for k, v in sig.parameters.items() if v.kind == 0}

        # check if method can be interpolated
        # currently only 1 or 2 positional parameters are allowed
        if len(sig.parameters) == 0:
            raise TypeError("only methods with at least 1 argument can be interpolated. "
                            f"{method_name} has 0.")
        elif len(sig.parameters) > 2:
            raise TypeError("only methods with 1 or 2 arguments can be interpolated. "
                            f"{method_name} has {len(sig.parameters)}.")

        # check that there is 1 interpkw for each positional parameter
        if interpkw and (len(interpkw) != len(pos_params)):
            raise ValueError("must specify interpolation arguments for each "
                             "positional parameter, missing "
                             f"{pos_params - interpkw.keys()}.")

        @functools.wraps(method)
        def wrapped(self, *args, uninterpolated=False):
            """Call interpolated method, or create if DNE.

            Parameters
            ----------
            *args
                Passed to underlying method.
            uninterpolated : bool
                Whether to call the interpolated or un-interpolated method.
            """
            print("here")
            if uninterpolated:
                return method(self, *args)
            elif method_name in self.__interpolated_methods__:
                return wrapped.__interpolated__(*args)
            # else:  # need to make interpolation. done below.

            # ---------------
            # Make and call interpolation

            interpkw.update({k: self._default_interpkw for k in pos_params
                             if k not in interpkw})

            # process parameters
            params = [None] * len(pos_params)
            for i, n in enumerate(pos_params):
                param_info = interpkw.pop(n)

                # TODO! allow less memory intensive controls.
                # eg generate z array here with (start, stop, num)
                z = param_info["range"]

                params[i] = z

            # call un-interpolated function over interpolant array and
            # create interpolation, type depends on number of variables
            if len(params) == 1:
                val = method(self, *params)
                interped = InterpolatedUnivariateSpline(params[0], val, **interpkw)
            else:  # len(params) == 2
                raise NotImplementedError("TODO!")
                # xg, yg = np.meshgrid(params[0], params[1], indexing='ij', sparse=True)
                # val = method(self, xg, yg)
                # interped = RectBivariateSpline(xg[0], yg[1], val, **interpkw)

            # store and register the interpolation function for re-use
            wrapped.__interpolated__ = interped
            self.__interpolated_methods__.add(method_name)

            # re-call method, now using the interpolation
            return wrapped(self, *args)

        # pre-load. overwritten in first call
        wrapped.__interpolated__ = None

        # set interpolation wrapper as method
        # can access original method with ``.__wrapped__``
        setattr(cls, method_name, wrapped)

    # ===============================================================

    def __init__(self, *args, **kwargs):
        # Interpolation attrs
        self._uninterpolated_ = None  # empty un-interpolated reference
        self.__interpolated_methods__ = set()  # interpolated methods
        self._default_interpkw = {}

        super().__init__(*args, **kwargs)

    @classmethod
    def register(cls, method=None, **interpkw):
        """

        Parameters
        ----------
        method : str or None
        **interpkw
            Keyword arguments for interpolation.
        """
        if cls is not Interpolated:
            raise TypeError

        if method is None:
            return functools.partial(cls.register, **interpkw)

        method.__interpinfo__ = interpkw
        return method


# Define the base of the interpolated-Cosmology class hierarchy
class InterpolatedCosmology(Cosmology, Interpolated, base_cls=Cosmology, data_cls=Cosmology):

    def __init_subclass__(cls, base_cls=None, data_cls=None, **kwargs):

        # Override __init__ to have the interpolation at the top of the MRO
        init = FunctionType(  # almost exact copy of __new__
            __init__.__code__, __init__.__globals__, name=__init__.__name__,
            argdefs=__init__.__defaults__, closure=__init__.__closure__)
        # TODO! properly mix in the signature
        # init = functools.update_wrapper(init, cls.__init__)  # update further
        init.__kwdefaults__ = cls.__init__.__kwdefaults__  # fill in kwdefaults
        cls.__init__ = init

        # TODO! register every non-registered redshift method
        # this overrides the default behavior of ``Interpolated``, which only
        # applies to specifically registered items.

        super().__init_subclass__(base_cls=base_cls, data_cls=data_cls, **kwargs)

    @property
    def uninterpolated(self):
        """Return the un-interpolated form of this cosmology."""
        # Make uninterpolated Cosmology if DNE
        if self._uninterpolated_ is None:
            # create arg/kwarg to initialize uninterpolated cosmology
            ba = self._init_signature.bind_partial(**self._init_arguments)
            # create instance from known un-interpolated base class
            cosmo = self._data_cls(*ba.args, **ba.kwargs)
            # on uninterpolated, store weakref to this interpolated instance
            cosmo._interpolated_ = weakref.ref(self)
            # and vice versa
            self._uninterpolated_ = weakref.ref(cosmo)

        return self._uninterpolated_()  # get instance from weakref

    def __init__(self, *args, zrange, **kwargs):
        # Interpolation attrs
        self._uninterpolated_ = None  # empty un-interpolated reference
        self.__interpolated_methods__ = set()  # interpolated methods
        self._default_interpkw = {"range": zrange}

        super(self.__class__, self).__init__(*args, **kwargs)

    @property
    @abc.abstractmethod
    def is_flat(self):
        """
        Return bool; `True` if the cosmology is flat.
        This is abstract and must be defined in subclasses.
        """
        super().is_flat()
