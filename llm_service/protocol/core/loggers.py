import logging
from llm_service.protocol.core.interfaces import Logger

class StandardLogger(Logger):
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)
    
    def info(self, msg: str, *args) -> None:
        self._logger.info(msg, *args)
        
    def warning(self, msg: str, *args) -> None:
        self._logger.warning(msg, *args)
    
    def error(self, msg: str, *args) -> None:
        self._logger.error(msg, *args)
    
    def exception(self, msg: str, *args) -> None:
        self._logger.exception(msg, *args)
        
    def debug(self, msg: str, *args) -> None:
        self._logger.debug(msg, *args)

class NoOpLogger(Logger):
    def info(self, msg: str, *args) -> None: pass
    def warning(self, msg: str, *args) -> None: pass
    def error(self, msg: str, *args) -> None: pass
    def exception(self, msg: str, *args) -> None: pass
    def debug(self, msg: str, *args) -> None: pass
