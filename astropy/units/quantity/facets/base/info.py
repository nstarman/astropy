# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
This module defines the `Quantity` object, which represents a number with some
associated units. `Quantity` objects support operations like ordinary numbers,
but will deal with unit conversions internally.
"""

from __future__ import annotations

from astropy.utils.data_info import ParentDtypeInfo

__all__ = ["QuantityInfoBase", "QuantityInfo"]


class QuantityInfoBase(ParentDtypeInfo):
    # This is on a base class rather than QuantityInfo directly, so that
    # it can be used for EarthLocationInfo yet make clear that that class
    # should not be considered a typical Quantity subclass by Table.
    attrs_from_parent = {'dtype', 'unit'}  # dtype and unit taken from parent
    _supports_indexing = True

    @staticmethod
    def default_format(val):
        return f'{val.value}'

    @staticmethod
    def possible_string_format_functions(format_):
        """Iterate through possible string-derived format functions.

        A string can either be a format specifier for the format built-in,
        a new-style format string, or an old-style format string.

        This method is overridden in order to suppress printing the unit
        in each row since it is already at the top in the column header.
        """
        yield lambda format_, val: format(val.value, format_)
        yield lambda format_, val: format_.format(val.value)
        yield lambda format_, val: format_ % val.value


class QuantityInfo(QuantityInfoBase):
    """
    Container for meta information like name, description, format.  This is
    required when the object is used as a mixin column within a table, but can
    be used as a general way to store meta information.
    """
    _represent_as_dict_attrs = ('value', 'unit')
    _construct_from_dict_args = ['value']
    _represent_as_dict_primary_data = 'value'

    def new_like(self, cols, length, metadata_conflicts='warn', name=None):
        """
        Return a new Quantity instance which is consistent with the
        input ``cols`` and has ``length`` rows.

        This is intended for creating an empty column object whose elements can
        be set in-place for table operations like join or vstack.

        Parameters
        ----------
        cols : list
            List of input columns
        length : int
            Length of the output column object
        metadata_conflicts : str ('warn'|'error'|'silent')
            How to handle metadata conflicts
        name : str
            Output column name

        Returns
        -------
        col : `~astropy.units.Quantity` (or subclass)
            Empty instance of this class consistent with ``cols``

        """

        # Get merged info attributes like shape, dtype, format, description, etc.
        attrs = self.merge_cols_attributes(cols, metadata_conflicts, name,
                                           ('meta', 'format', 'description'))

        # Make an empty quantity using the unit of the last one.
        shape = (length,) + attrs.pop('shape')
        dtype = attrs.pop('dtype')
        # Use zeros so we do not get problems for Quantity subclasses such
        # as Longitude and Latitude, which cannot take arbitrary values.
        data = np.zeros(shape=shape, dtype=dtype)
        # Get arguments needed to reconstruct class
        map = {key: (data if key == 'value' else getattr(cols[-1], key))
               for key in self._represent_as_dict_attrs}
        map['copy'] = False
        out = self._construct_from_dict(map)

        # Set remaining info attributes
        for attr, value in attrs.items():
            setattr(out.info, attr, value)

        return out

    def get_sortable_arrays(self):
        """
        Return a list of arrays which can be lexically sorted to represent
        the order of the parent column.

        For Quantity this is just the quantity itself.


        Returns
        -------
        arrays : list of ndarray
        """
        return [self._parent]
