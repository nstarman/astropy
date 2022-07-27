#!/usr/bin/env python
# Licensed under a 3-clause BSD style license - see LICENSE.rst

# NOTE: The configuration for the package, including the name, version, and
# other information are set in the setup.cfg file.

import pathlib
import os
from setuptools import setup
from extension_helpers import get_extensions

setup(
    use_scm_version=True,
    ext_modules=get_extensions()
 )
