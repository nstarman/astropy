# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.units import Quantity


class PrivateParameter:
    # TODO! it might be faster for the PrivateParameter to be stored in `_[name]_`,
    # not `_[name]` and have the actual value be moved to `_[name]`, not
    # `__dict__[name]`

    def __init__(self, name):
        self._attr_name = name
        self._private_name = "_" + name
        self._value_name = "_" + name + "_value"

    def __get__(self, instance, owner=None):
        # only works on instances
        try:
            return instance.__dict__[self._private_name]
        except KeyError:
            raise AttributeError  # TODO! message

    def __set__(self, instance, value):
        if self._attr_name in instance.__parameters__:  # then invoke 'make'
            value = getattr(instance.__class__, self._attr_name).make(value)

        if value is None:
            instance.__dict__[self._private_name] = None
            return

        value = np.asanyarray(value)  # ensure numpy viewable object
        instance.__dict__[self._private_name] = value
        instance.__dict__[self._value_name] = value.view(np.float64, np.ndarray)


class Parameter(property):  # TODO! try using some stuff in 'property'

    def __init__(self, default=None, unit=None, fget=None, doc=None):
        super().__init__(fget=fget, doc=doc)
        self._default = default
        self._unit = unit
        # self.__doc__ = __doc__

    def __set_name__(self, owner, name):
        self._private_name = "_" + name

    @property
    def default(self):
        return self._default

    @property
    def unit(self):
        return self._unit

    def make(self, value):
        if value is None:
            value = self._default
        if self.unit is not None:
            value = Quantity(value, self.unit)

        if issubclass(type(value), np.ndarray):
            value.flags.writeable = False  # read-only

        return value

    # -------------------------------------------

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        if self.fget is None:
            # return instance.__dict__[self._private_name]
            return getattr(instance, self._private_name)
        return self.fget(instance)

    def getter(self, fget):
        return type(self)(self.default, self.unit, fget=fget, doc=self.__doc__)
