# Makes 'tether_service.config' a proper package for importlib.resources.
from tether_service.config._strict import StrictModel
from tether_service.config.settings import Settings, load_settings

__all__ = ["Settings", "StrictModel", "load_settings"]