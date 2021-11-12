import os
import sys

sys.path.insert(0, os.path.dirname(__file__))  # allows import of "mypackage"

# isort: split
import mypackage

from . import test_connect, test_core, test_flrw, test_parameter
