# -*- coding: utf-8 -*-

"""Typing compatibility for :mod:`~astropy.units`.

Static types are imported from

1. ``typing`` (if present)
2. ``typing_extensions`` (if present)
3. Here if not present in the above.
   These are altered from https://github.com/python/typing

"""

__all__ = [
    "Annotated",
]

import typing
import operator

try:  # py 3.9+
    from typing import Annotated
except ImportError:  # optional dependency
    try:
        from typing_extensions import Annotated
    except ImportError:
        _HAS_ANNOTATED = False   # need to do ourselves
    else:
        _HAS_ANNOTATED = True
else:
    _HAS_ANNOTATED = True


NoneType = type(None)


def _isAnnotated(t) -> bool:
    """Is this thing an annotated type?

    Checks that the object has attributes origin and metadata
    and that it passes an equality test with Annotated[origin, metadata].
    This test is designed to pass if ``Annotated`` is from
    :mod:`~typing`, ``typing_extensions``, or our compatibility class.

    .. todo::

        When min python version is 3.9+ or typing_extensions is a run-time
        dependency, I think there's a built-in method to test if something
        is an Annotated.

    Parameters
    ----------
    t : object

    Returns
    -------
    bool
        Whether the object is an ``Annotated``.

    """
    if not hasattr(t, "__origin__") or not hasattr(t, "__metadata__"):
        return False

    # we need to construct the "expected" Annotation with the underlying
    # class getitem to avoid having to special-case single annotations.
    # Annotated turns annotations into a tuple, so we want to ensure
    # correct unpacking, which can only be done with the function syntax,
    # not the getitem [].
    expected = Annotated.__class_getitem__((t.__origin__, *t.__metadata__))

    if t == expected:
        return True

    return False


def _isUnion(t) -> bool:
    """Check whether a type is a Union."""
    if hasattr(t, "__origin__"):
        if t.__origin__ is typing.Union:
            return True

    return False


#####################################################################

if not _HAS_ANNOTATED:

    try:
        from typing import _tp_cache
    except ImportError:

        def _tp_cache(x):
            return x

    class _AnnotatedAlias(typing._GenericAlias, _root=True):
        """Runtime representation of an annotated type.

        At its core 'Annotated[t, dec1, dec2, ...]' is an alias for the
        type 't' with extra annotations. The alias behaves like a normal
        typing alias, instantiating is the same as instantiating the
        underlying type, binding it to types is also the same.

        """

        def __init__(self, origin, metadata):
            if isinstance(origin, _AnnotatedAlias):
                metadata = origin.__metadata__ + metadata
                origin = origin.__origin__
            super().__init__(origin, origin)
            self.__metadata__ = metadata

        def copy_with(self, params):
            assert len(params) == 1
            new_type = params[0]
            return _AnnotatedAlias(new_type, self.__metadata__)

        def __repr__(self):
            return "Annotated[{}, {}]".format(
                typing._type_repr(self.__origin__),
                ", ".join(repr(a) for a in self.__metadata__),
            )

        def __reduce__(self):
            return (
                operator.getitem,
                (Annotated, (self.__origin__,) + self.__metadata__),
            )

        def __eq__(self, other):
            if not isinstance(other, _AnnotatedAlias):
                return NotImplemented
            if self.__origin__ != other.__origin__:
                return False
            return self.__metadata__ == other.__metadata__

        def __hash__(self):
            return hash((self.__origin__, self.__metadata__))

    class Annotated:
        """Add context specific metadata to a type.

        Example: Annotated[int, runtime_check.Unsigned] indicates to the
        hypothetical runtime_check module that this type is an unsigned int.
        Every other consumer of this type can ignore this metadata and treat
        this type as int.

        The first argument to Annotated must be a valid type (and will be in
        the __origin__ field), the remaining arguments are kept as a tuple in
        the __extra__ field.

        Details:

        - It's an error to call `Annotated` with less than two arguments.
        - Nested Annotated are flattened::

            Annotated[Annotated[T, Ann1, Ann2], Ann3] == Annotated[T, Ann1, Ann2, Ann3]

        - Instantiating an annotated type is equivalent to instantiating the
        underlying type::

            Annotated[C, Ann1](5) == C(5)

        - Annotated can be used as a generic type alias::

            Optimized = Annotated[T, runtime.Optimize()]
            Optimized[int] == Annotated[int, runtime.Optimize()]

            OptimizedList = Annotated[List[T], runtime.Optimize()]
            OptimizedList[int] == Annotated[List[int], runtime.Optimize()]
        """

        __slots__ = ()

        def __new__(cls, *args, **kwargs):
            raise TypeError("Type Annotated cannot be instantiated.")

        @_tp_cache
        def __class_getitem__(cls, params):
            if not isinstance(params, tuple) or len(params) < 2:
                raise TypeError(
                    "Annotated[...] should be used "
                    "with at least two arguments (a type and an "
                    "annotation)."
                )
            msg = "Annotated[t, ...]: t must be a type."
            origin = typing._type_check(params[0], msg)
            metadata = tuple(params[1:])
            return _AnnotatedAlias(origin, metadata)

        def __init_subclass__(cls, *args, **kwargs):
            raise TypeError(
                "Cannot subclass {}.Annotated".format(cls.__module__)
            )
