# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Test utilities for :mod:`astropy.units._typing`.

These tests are skipped if :mod:`~typing` or ``typing_extensions```
have type ``Annotated``.

"""

import pytest

try:  # py3.9+ builtin
    from typing import Annotated
except ImportError:
    try:  # optional dependency
        from typing_extensions import Annotated
    except ImportError:  # need to test
        from astropy.units._typing import Annotated, _isAnnotated
    else:
        pytest.skip("typing_extensions optional dependency",
                    allow_module_level=True)

else:
    pytest.skip("Python built-in", allow_module_level=True)


def _assert_an_annotation(annot, origin, metadata):
    assert _isAnnotated(annot), annot
    assert annot.__args__[0] == origin
    assert annot.__metadata__ == (metadata,)


def test_Annotated():
    """Test :class:`~astropy.untis._typing.Annotated`,

    If either ``typing.Annotated`` or ``typing_extensions.Annotated``
    are available, this is test is NOT run.

    """
    # ----------------------------
    # Test Fails
    # Cannot instantiate Annotated
    with pytest.raises(TypeError):
        Annotated()

    # Cannot subclass Annotated
    with pytest.raises(TypeError):

        class AnnotatedSubclass(Annotated):
            pass

    # Annotated 1 annotation
    with pytest.raises(TypeError):
        Annotated[tuple]

    # ----------------------------
    # Test Usage
    # Instantiation
    annot = Annotated[tuple, "metadata"]
    _assert_an_annotation(annot, tuple, "metadata")

    # field test on a function
    def func(x: annot) -> annot:
        pass

    x_annot = func.__annotations__["x"]
    _assert_an_annotation(x_annot, tuple, "metadata")

    return_annot = func.__annotations__["return"]
    _assert_an_annotation(return_annot, tuple, "metadata")


# def test_AnnotatedAlias():
#     """
#         Test :class:`~astropy.untis._typing._AnnotatedAlias`,
#         if neither :mod:`~typing` nor ``typing_extensions```
#         have a class ``Annotated``.

#     """
#     annot = _AnnotatedAlias(tuple, "metadata")

#     annot2 = _AnnotatedAlias(annot, ("metadata 2",))
#     assert isinstance(annot2, _AnnotatedAlias)
#     assert annot2.__origin__ == tuple
#     assert annot2.__metadata__ == ("metadata", "metadata 2")

#     # copy_with
#     annot3 = annot.copy_with((list, ))
#     assert annot3.__origin__ == list

#     # __repr__
#     assert annot.__repr__() == "Annotated[{}, {}]".format(
#         annot.__origin__, ", ".join(repr(a) for a in annot.__metadata__)
#     )

#     # __reduce__
#     assert annot.__reduce__() == (
#         operator.getitem,
#         (Annotated, (annot.__origin__,) + annot.__metadata__),
#     )

#     # __eq__
#     assert (annot == None) == NotImplemented  # noqa: E711  # different type
#     assert (annot == annot3) is False  # different origin
#     assert (annot == annot) is True  # same origin & metadata
#     assert (annot == annot2) is False  # same origin, different metadata

#     # __hash__
#     assert annot.__hash__() == hash((annot.__origin__, annot.__metadata__))
