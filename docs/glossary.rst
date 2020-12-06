.. currentmodule:: astropy

*************
Term Glossary
*************

.. glossary::

   Unit-like
      Must be an `~astropy.units.UnitBase` instance or a string or other instance
      parseable by `~astropy.units.Unit`.

   Quantity-like
      Must be an `~astropy.units.Quantity` (or subclass) instance or a string
      parseable by `~astropy.units.Quantity`. Note that the interpretation of
      units in strings depends on the class: ``Quantity("180d")`` is 180 days,
      while ``Angle("180d")`` is 180 degrees.

   angular-Quantity-like
      :term:`Quantity-like`, but interpreted by an angular
      :`~astropy.units.SpecificTypeQuantity`, like
      :`~astropy.coordinates.Angle` or ~astropy.coordinates.Longitude` or
      :~astropy.coordinates.Latitude`.

   distance Quantity-like
      :term:`Quantity-like`, but interpretable by :~astropy.coordinates.Distance`.

   frame-like
      a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance
      or a :class:`~astropy.coordinates.SkyCoord` (or subclass) instance or a
      string that can be converted to a Frame by
      :class:`~astropy.coordinates.sky_coordinate_parsers._get_frame_class`.

   coord-like
      a :class:`~astropy.coordinates.BaseCoordinateFrame` subclass instance or a
      :class:`~astropy.coordinates.SkyCoord` (or subclass) instance.

   coord-like scalar
   coord scalar
      a :term:`coord-like` object with length 1.

   coord-like array
   coord array
      a :term:`coord-like` object with length > 1.

   ColDefs-like
      A column-like structure from which a `~astropy.io.fits.column.ColDefs`
      can be instantiated. This includes an existing
      `~astropy.io.fits.hdu.table.BinTableHDU` or
      `~astropy.io.fits.hdu.table.TableHDU`, or a `numpy.recarray` to give
      some examples.

   Time-like
      `~astropy.time.Time` or any valid initializer.

   number
      Any numeric type. eg float or int.
