# -*- coding: utf-8 -*-

"""Fields on Representations.

Example

.. code-block:: python

    from astropy import table, coordinates as coord, units as u
    import numpy as np

    points = coord.CartesianRepresentation(x=np.linspace(0, 10) * u.kpc,
                                           y=np.linspace(-1, -11) * u.kpc,
                                           z=np.linspace(-3, 3) * u.kpc)
    fr = coord.SphericalFieldRepresentation(np.linspace(0, 10) * u.km / u.s**2,
                                            np.linspace(10, 100) * u.km / u.s**2,
                                            np.linspace(100, 1000) * u.km / u.s**2,
                                            points=points)
    fr[:3]

    field = coord.Field(
        coord.CartesianRepresentation(x=np.linspace(0, 10) * u.kpc,
                                      y=np.linspace(-1, -11) * u.kpc,
                                      z=np.linspace(-3, 3) * u.kpc),
        mass=1 * u.solMass,
        vector=np.c_[np.linspace(0, 10),
                       np.linspace(10, 100),
                       np.linspace(100, 1000)] * u.km / u.s**2,)
    
    field.points[:4]

    field.vector[:4]

    field.fieldtypes

    field.represent_as("spherical").vector[:3]

    field._data


"""

__all__ = [
    "Field",
    # Representations
    "CartesianFieldRepresentation",
    "SphericalFieldRepresentation",
    "PhysicsSphericalFieldRepresentation",
    "CylindricalFieldRepresentation",
]


##############################################################################
# IMPORTS

# BUILT-IN
import copy
import inspect
import functools
import operator
from types import MappingProxyType

# THIRD PARTY
import numpy as np
from erfa import ufunc as erfa_ufunc

# PROJECT-SPECIFIC
import astropy.units as u
from astropy import table

from .representation import (
    _array2string, _make_getter, REPRESENTATION_CLASSES,
    BaseRepresentation, CartesianRepresentation, SphericalRepresentation,
    PhysicsSphericalRepresentation, CylindricalRepresentation)


##############################################################################
# PARAMETERS

_REPRESENTATION_TO_FIELD_MAPPING = dict()

##############################################################################
# CODE
##############################################################################

def _make_point_getter(component):
    def getter(self):
        return getattr(self._points, component)
    
    return getter

class FieldRepresentationBase(BaseRepresentation):
    """Representation of Components of a Field.

    Parameters
    ----------
    *args
        The components. All components must have the same units.
    points : `~astropy.coordinates.BaseRepresentation`
        The points of the field.

    """
 
    attr_classes = {}  # needed b/c of BaseRepresentation meta

    def __init_subclass__(cls):
        # discover the base representation from the MRO
        base_rep = [kls for kls in cls.mro() if ("field" not in kls.__name__.lower())][0]
        # store here and in lookup-mapping for later use
        cls.base_representation = base_rep
        _REPRESENTATION_TO_FIELD_MAPPING[base_rep] = cls

        # an invariant set of component names
        cls.attr_classes = {f"q{i+1}": u.Quantity for i in
                            range(len(base_rep.attr_classes.keys()))}

        # add properties to access the field and points components
        for i, k in enumerate(base_rep.attr_classes):
            # a more descriptive name than q1
            setattr(cls, f"q_{k}",
                    property(_make_getter(f"q{i+1}"),
                             doc=f"Component q{i} ({k}) of the field."))

            # access to the underlying points' components
            # TODDO a la _make_getter this
            setattr(cls, k,
                    property(_make_point_getter(k),
                             doc=f"Component q{i+1} ({k}) of the field."))

    def __init__(self, *args, points, copy=True):
        # store the points, in the correct representation
        self._points = points.represent_as(self.base_representation)
        # cast to Quantity. also does equal units check.
        args = u.Quantity(args, copy=False)
        # make single input interpretable
        if len(args) == 1:
            args = args[0].T
        # store each component
        self._q1 = args[0]
        self._q2 = args[1]
        self._q3 = args[2]
        # there can be no differentials
        self._differentials = MappingProxyType({})

    @property
    def points(self):
        """The points at which the field values are defined."""
        return self._points

    @property
    def q1(self):
        return self._q1

    @property
    def q2(self):
        return self._q2

    @property
    def q3(self):
        return self._q3

    # =====================================================
    # Representation Conversion

    def scale_factors(self):
        r"""Fields have scale factors of 1."""
        l = np.broadcast_to(1.*u.one, self.shape, subok=True)
        return {k: l for k in self.attr_classes.keys()}

    def unit_vectors(self):
        r"""Cartesian unit vectors in the direction of each component.

        Given unit vectors :math:`\hat{e}_c`, a change in one component of
        :math:`\delta c` corresponds to a change in representation of
        :math:`\delta c \times \hat{e}_c`.

        Returns
        -------
        unit_vectors : dict of `CartesianRepresentation`
            The keys are the component names.

        """
        return self.points.unit_vectors()

    def to_cartesian(self):
        """Convert the field to 3D rectangular Cartesian coordinates.

        Returns
        -------
        `CartesianFieldRepresentation`
            This object, converted.

        """
        base_e = self.unit_vectors()  # points' unit vectors
        c = functools.reduce(  # dot each component
            operator.add,
            (getattr(self, f) * base_e[c]
             for f, c in zip(self.components, self.points.components)))

        return CartesianFieldRepresentation(c.x, c.y, c.z,
                                            points=self.points.to_cartesian())

    @classmethod
    def from_cartesian(cls, other):
        """Convert field from 3D Cartesian to the desired representation type.

        Parameters
        ----------
        other : `CartesianFieldRepresentation`
            The object to convert into this field.

        Returns
        -------
        `~astropy.coordinates.FieldRepresentationBase` subclass instance
            A new field object that is this class' type.

        """
        # convert points to the rep type for this field
        points = cls.base_representation.from_cartesian(other.points)
        base_e = points.unit_vectors()  # conversion factors for field

        return cls(*(other.dot(e) for e in base_e.values()), points=points,
                   copy=False)

    def represent_as(self, other_class):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via Cartesian coordinates.

        Parameters
        ----------
        other_class : `~FieldRepresentationBase` subclass, `~BaseRepresentation` subclass, or str
            The type of representation to turn the field into.
            If this is a string, it must be a valid Representation name.

        Returns
        -------
        `~FieldRepresentationBase` subclass instance

        """
        if other_class is self.__class__:  # shortcut if no change needed
            return self

        # process the other_class options
        if inspect.isclass(other_class) and issubclass(other_class,FieldRepresentationBase,):
            pass
        elif inspect.isclass(other_class) and issubclass(other_class, BaseRepresentation):
            other_class = _REPRESENTATION_TO_FIELD_MAPPING[other_class]
        elif isinstance(other_class, str):
            rep_cls = REPRESENTATION_CLASSES[other_class]
            other_class = _REPRESENTATION_TO_FIELD_MAPPING[rep_cls]
        else:
            raise TypeError

        # The default is to convert via cartesian coordinates.
        # subclasses should override.
        self_cartesian = self.to_cartesian()
        return other_class.from_cartesian(self_cartesian)

    # =====================================================
    # Math

#     def _scale_operation(self, op, *args):
#         """Scale all components.
# 
#         Parameters
#         ----------
#         op : `~operator` callable
#             Operator to apply (e.g., `~operator.mul`, `~operator.neg`, etc.
#         *args
#             Any arguments required for the operator (typically, what is to
#             be multiplied with, divided by).
# 
#         """
#         scaled_attrs = [op(getattr(self, c), *args) for c in self.components]
#         scaled_points = self.points._scale_operation(op, *args)
#         return self.__class__(
#             *scaled_attrs,
#             copy=False,
#             points=scaled_points,
#         )
    
    def _combine_operation(self, op, other, reverse = False):
        diff = (self.points.represent_as(CartesianRepresentation)
                - other.points.represent_as(CartesianRepresentation))
        if not np.allclose(diff.norm().value, 0):
            raise Exception("can't combine mismatching points.")

        # ----------

        if isinstance(self, type(other)):
            first, second = (self, other) if not reverse else (other, self)
            return self.__class__(
                *[
                    op(getattr(first, c), getattr(second, c))
                    for c in self.components
                ],
                points=self.points,
            )
        else:
            try:
                self_cartesian = self.to_cartesian()
            except TypeError:
                return NotImplemented

            return other._combine_operation(op, self_cartesian, not reverse)

    def norm(self):
        return np.sqrt(functools.reduce(
                operator.add,
                (getattr(self, component) ** 2 for component, cls in self.attr_classes.items())))

    def _apply(self, method, *args, **kwargs):
        if callable(method):

            def apply_method(array):
                return method(array, *args, **kwargs)

        else:
            apply_method = operator.methodcaller(method, *args, **kwargs)

        new = super().__new__(self.__class__)
        new._points = self.points._apply(method, *args, **kwargs)
        for component in self.components:
            setattr(
                new,
                "_" + component,
                apply_method(getattr(self, component)),
            )

        # Copy other 'info' attr only if it has actually been defined.
        # See PR #3898 for further explanation and justification, along
        # with Quantity.__array_finalize__
        if "info" in self.__dict__:
            new.info = self.info

        return new

    # =====================================================
    # Misc

    def __repr__(self) -> str:
        prefixstr = "    "
        # TODO combine with points
        arrstr = _array2string(
            np.lib.recfunctions.merge_arrays(
                (self.points._values, self._values),
            ),
            prefix=prefixstr,
        )

        pointsunitstr = (
            ("in " + self.points._unitstr)
            if self.points._unitstr
            else "[dimensionless]"
        )
        unitstr = (
            ("in " + self._unitstr) if self._unitstr else "[dimensionless]"
        )
        return "<{} ({}) {:s} | ({}) {:s}\n{}{}>".format(
            self.__class__.__name__,
            ", ".join(self.points.components),
            pointsunitstr,
            ", ".join(self.components),
            unitstr,
            prefixstr,
            arrstr,
        )


# -------------------------------------------------------------------


class CartesianFieldRepresentation(FieldRepresentationBase,
                                   CartesianRepresentation):

    def get_xyz(self, xyz_axis=0):
        return np.stack([self._q1, self._q2, self._q3], axis=xyz_axis)

    def dot(self, other):
        other_c = other.to_cartesian()
        other_xyz = other_c.get_xyz(xyz_axis=-1)
        return erfa_ufunc.pdp(self.get_xyz(xyz_axis=-1), other_xyz)


# -------------------------------------------------------------------

class SphericalFieldRepresentation(FieldRepresentationBase,
                                   SphericalRepresentation):
    pass


# -------------------------------------------------------------------

class PhysicsSphericalFieldRepresentation(FieldRepresentationBase,
                                          PhysicsSphericalRepresentation):
    pass


# -------------------------------------------------------------------

class CylindricalFieldRepresentation(FieldRepresentationBase,
                                     CylindricalRepresentation):
    pass


##############################################################################
    
class Field:
    
    # =====================================================
    # Properties

    @property
    def meta(self):
        return self._data.meta

    @property
    def fieldtypes(self):
        return MappingProxyType(self._data.meta["__fieldtype__"])

    @property
    def points(self):
        return self._data["coord"]

    # =====================================================
    # Initialization

    def __new__(cls, points, *args, **kwargs):
        if isinstance(points, table.Table):
            return cls.from_table(points)
        return super().__new__(cls)

    def __init__(self, points, **fields):
        self._data = table.QTable()
        self._data["coord"] = points

        self.meta["__fieldtype__"] = dict()
        for k, v in fields.items():
            self[k] = v

    @classmethod
    def from_table(cls, tbl, copy=True):
        # TODO! validation
        self = super().__new__(cls)
        self._data = table.QTable(tbl, copy=copy)
        return self

    # =====================================================

    def __getattr__(self, key):
        """Map ``getattr`` to columns on the underlying data table."""
        return self._data[key]

    def __getitem__(self, key):
        """Get columns from the underlying data table."""
        return self._data[key]

    def __setitem__(self, key, value):
        # TODO! validation and stuff
        fieldtype = "scalar" if not np.shape(value) else "vector"
        if fieldtype == "vector":
            vf_cls = _REPRESENTATION_TO_FIELD_MAPPING[self.points.__class__]
            value = vf_cls(value, points=self.points)  # TODO! ensure always a view

        self.meta["__fieldtype__"][key] = fieldtype
        self._data[key] = value

    # =====================================================
    # Representation Conversion

    def represent_as(self, representation_type, inplace=True):
        if not inplace:
            self = copy.deepcopy(self)

        # convert str to class
        if isinstance(representation_type, str):
            representation_type = REPRESENTATION_CLASSES[representation_type]

        self._data["coord"] = self._data["coord"].represent_as(representation_type)
        for k in (k for k, v in self.fieldtypes.items() if v == "vector"):
            self._data[k] = self._data[k].represent_as(representation_type)

        return self

    # =====================================================
    # Misc

    def __repr__(self):
        return repr(self._data).replace("QTable", "Field")


##############################################################################
# END
