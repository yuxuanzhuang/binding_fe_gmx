from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .awh import AWH_Ensemble, AWH_2D_Ensemble
