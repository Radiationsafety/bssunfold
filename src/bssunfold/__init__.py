# bssunfold/__init__.py
import importlib.metadata
__all__ = ["Detector", "ICRP116_COEFF_EFFECTIVE_DOSE", "RF_GSF"]

from .detector import Detector
from .constants import ICRP116_COEFF_EFFECTIVE_DOSE, RF_GSF

try:
    __version__ = importlib.metadata.version("bssunfold")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"
