Added a registry of cosmology realizations to ``default_cosmology``.
Realizations can be added to the ``default_cosmology`` registry with ``register``
and then used in the ``get`` / ``set`` methods by name.
Also, the ``get_cosmology_from_string`` method is deprecated and consolidated
into ``get``, which now takes an optional argument to get a specific realization,
not the current ScienceState value.