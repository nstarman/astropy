Unit-Aware Type Annotations
***************************

.. |Quantity| replace:: :class:`~astropy.units.Quantity`

While Python is a dynamically typed language, there is also support for type hints (and even static typing) using the annotations syntax of `PEP 484 <https://www.python.org/dev/peps/pep-0484/>`_.
For a detailed guide on type hints, function annotations, and other related syntax see the `Real Python Guide <https://realpython.com/python-type-checking/#type-aliases>`_. 
Here we how Astropy |Quantity| can be used in type hints and annotations and also include metadata about the associated units.


For this and all further examples we assume the following imports

::

   >>> import typing as T
   >>> import astropy.units as u
   >>> from astropy.units import Quantity
   >>> from astropy.units._typing import Annotated


.. _quantity_type_annotation:

Quantity Type Annotation
========================

A |Quantity| can easily be used as a type annotation,

::

   >>> x: Quantity = 2 * u.km

or as a function annotation.

::

   >>> def func(x: Quantity) -> Quantity:
   ...     return x

However, depending on the computer, this frequently throws an exception when compiling Sphinx docs. A safer method to annotate Quantities is by leveraging the :mod:`~typing` library to create a type variable (note the ``~``).

::

   >>> QuantityType = T.TypeVar("QuantityType", bound=Quantity)

Anything annotated with ``QuantityType`` is expected to be a |Quantity| or subclass thereof.

Repeating the above examples,

::

   >>> x: QuantityType = 2 * u.km

or as a function annotation.

::

   >>> def func(x: QuantityType) -> QuantityType:
   ...     return x


Preserving Units
^^^^^^^^^^^^^^^^

While the above annotations are useful for annotating the **value**'s type, it does not inform us of the other most important attribute of a |Quantity|: the **unit**.

Unit information may be included by the syntax ``Quantity[unit or "physical_type", other_meta]``. **The unit metadata must be first.**

::

   >>> Quantity[u.m]
   Annotated[~<class 'astropy.units.quantity.Quantity'>, Unit("m")]

   >>> Quantity["length"]
   Annotated[~<class 'astropy.units.quantity.Quantity'>, 'length']

   >>> Quantity[u.m, "Call", "me", "Ishmael."]
   Annotated[~<class 'astropy.units.quantity.Quantity'>, Unit("m"), 'Call', 'me', 'Ishmael.']

See :ref:`manual_annotation` for explanation of ``Annotated``

These can also be used on functions

::

   >>> def func(x: Quantity[u.kpc, "input"]) -> Quantity[u.m, "output"]:
   ...     return x << u.m


.. _manual_annotation:

Manually Constructing an Annotation
===================================

The ``Quantity[unit or "physical_type", other_meta]`` annotations can also be manually constructed. While this is not recommended, we show it here to explain the conventions and how it relates to the python :mod:`typing` library.

Unit-aware |Quantity| annotations are an annotation to the type using the ``Annotated`` class.

In Python 3.9+ ``Annotated`` is built into
:mod:`~typing`. For older python versions, the optional dependency
``typing_extensions`` provides the same class. When neither are
available we default to a run-time only compatibility class (in
``astropy.units._typing``) with reduced functionality. The best
implementation is automatically detected and used when importing ``Annotated`` from ``astropy.units._typing``.

::

  >>> Annotated[QuantityType, u.m, "I am a leaf on the wind"]
  Annotated[~QuantityType, Unit("m"), 'I am a leaf on the wind']

**The unit metadata must be first.**
``Annotated`` supports an arbitrary amount of metadata, but Astropy will only use the first for units.
Other metadata might still be useful, for instance as a description of the variable.

::

   >>> annot = Annotated[QuantityType, u.m, "square dimension"]
   >>> x: annot = 1 * u.m
   >>> y: annot = 2 * u.m


.. _multiple_annotation:

Multiple Annotations
====================

Multiple Quantity and unit-aware |Quantity| annotations are supported using :class:`~typing.Union`

::

   >>> T.Union[Quantity[u.m], None]
   typing.Union[Annotated[~<class 'astropy.units.quantity.Quantity'>, Unit("m")], NoneType]

::

   >>> T.Union[Quantity[u.m], Quantity["time"]]
   typing.Union[Annotated[~<class 'astropy.units.quantity.Quantity'>, Unit("m")], Annotated[~<class 'astropy.units.quantity.Quantity'>, 'time']]
