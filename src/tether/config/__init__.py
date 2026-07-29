# Makes 'tether.config' a proper package for importlib.resources.
from tether.config._strict import StrictModel
from tether.config.settings import Settings, load_settings

__all__ = ["Settings", "StrictModel", "load_settings"]
