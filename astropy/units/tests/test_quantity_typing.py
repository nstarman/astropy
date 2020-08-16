# coding: utf-8
# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Test the Quantity class and related."""

import sys
import typing as T

import pytest

from astropy import units as u

""" The Quantity class will represent a number + unit + uncertainty """


class TestQuantityTyping:
    """Test Quantity Typing Annotations.

    .. todo::

        - switch tests to ``get_args``, ``get_origin`` when min python 3.8+

    """

    def test_quantity_typing(self):
        """Test type hint creation from Quantity."""
        annot = u.Quantity[u.m]

        assert annot.__args__[0].__class__ is T.TypeVar
        assert annot.__args__[0].__bound__ is u.Quantity
        assert annot.__metadata__ == (u.m,)

        # test usage
        def func(x: annot, y: str) -> u.Quantity[u.s]:
            return x, y

        annots = T.get_type_hints(func)
        assert annots["x"] == annot
        assert annots["return"].__metadata__[0] == u.s

        # --------------------
        # Multiple Annotated

        multi_annot = u.Quantity[u.m, "other"]

        def multi_func(x: multi_annot, y: str):
            return x, y

        annots = T.get_type_hints(multi_func)
        assert annots["x"] == multi_annot

        # --------------------
        # Optional and Annotated

        opt_annot = T.Optional[u.Quantity[u.m]]

        def opt_func(x: opt_annot, y: str):
            return x, y

        annots = T.get_type_hints(opt_func)
        assert annots["x"] == opt_annot

        # --------------------
        # Union and Annotated

        # double Quantity[]
        union_annot1 = T.Union[u.Quantity[u.m], u.Quantity[u.s]]
        # one Quantity, one physical-type
        union_annot2 = T.Union[u.Quantity[u.m], u.Quantity["time"]]
        # one Quantity, one general type
        union_annot3 = T.Union[u.Quantity[u.m / u.s], float]

        def union_func(x: union_annot1, y: union_annot2) -> union_annot3:
            if isinstance(y, str):  # value = time
                return x.value  # returns <float>
            else:
                return x / y  # returns Quantity[m / s]

        annots = T.get_type_hints(union_func)
        assert annots["x"] == union_annot1
        assert annots["y"] == union_annot2
        assert annots["return"] == union_annot3

    def test_quantity_subclass_typing(self):
        """Test type hint creation from a Quantity subclasses."""

        class Length(u.SpecificTypeQuantity):
            _equivalent_unit = u.m

        annot = Length[u.km]

        assert annot.__args__[0].__class__ is T.TypeVar
        assert annot.__args__[0].__bound__ is Length
        assert annot.__metadata__ == (u.km,)

    def test_parse_allowed_type_hint(self):
        """Test helper function ``_parse_allowed_type_hint``."""
        assert u.quantity._parse_allowed_type_hint(u.m) == u.m
        assert u.quantity._parse_allowed_type_hint("length") == "length"

        # test a failure  # TODO test failure detail level.
        with pytest.raises(ValueError):
            u.quantity._parse_allowed_type_hint("will_fail")

        # test a failure, and suppressing detail
        with pytest.raises(ValueError):
            u.quantity._parse_allowed_type_hint(
                "will_fail", detailed_exception=False
            )
