from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

from .awh import (AWH_Ensemble,
                  AWH_1D_Ensemble,
                  AWH_2D_Ensemble,
                  AWH_3D_Ensemble
                    )
