import json
import re
from typing import Any, Dict, Optional
from llm_service.protocol.core.interfaces import ArgsParser, Logger
from llm_service.protocol.core.loggers import NoOpLogger

class JsonArgsParser(ArgsParser):
    """
    Parses JSON arguments for tool calls.
    """
    def __init__(self, logger: Optional[Logger] = None):
        self._logger = logger or NoOpLogger()

    def parse(self, raw_args: str) -> Dict[str, Any]:
        """
        Parse a raw string containing JSON arguments into a dictionary.
        Handles empty arguments and malformed JSON with multiple fallback strategies.
        """
        self._logger.info("==== PARSING TOOL ARGUMENTS ====")
        self._logger.info("Raw args: %r", raw_args)

        # Handle empty arguments case
        if not raw_args or raw_args.strip() == "":
            self._logger.info("Empty arguments provided, returning empty dict")
            return {}

        # Check for common formatting issues and fallbacks
        try:
            # If looks like JSON object, try loading
            if raw_args.strip().startswith('{'):
                result = json.loads(raw_args)
                self._logger.info("Successfully parsed JSON args: %s", result)
                return result
        except json.JSONDecodeError:
            self._logger.warning("JSON decode failed, trying fallback formats")

        # Fallback: simple key=value pairs
        fallback: Dict[str, Any] = {}
        for part in re.split(r'\s*,\s*', raw_args.strip(' {}"')):
            if '=' in part:
                key, val = part.split('=', 1)
                fallback[key.strip()] = val.strip().strip('"\'')
        if fallback:
            self._logger.info("Fallback parsed args: %s", fallback)
            return fallback

        # Last resort: return raw in error field
        self._logger.error("Unable to parse arguments, returning raw string")
        return {"_raw": raw_args}
