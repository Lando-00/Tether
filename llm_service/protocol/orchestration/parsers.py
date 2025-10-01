import json
from typing import Any, Dict, Optional, Union
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
        Handles empty arguments and malformed JSON.
        """
        if not raw_args or raw_args.strip() == "":
            self._logger.debug("Empty arguments provided")
            return {}
        
        try:
            return json.loads(raw_args)
        except json.JSONDecodeError as e:
            self._logger.warning(f"Invalid JSON arguments: {e}")
            
            # Attempt basic key-value string parsing as fallback
            try:
                return self._fallback_parse(raw_args)
            except Exception as parse_error:
                self._logger.error(f"Fallback parsing failed: {parse_error}")
                return {"_error": f"Invalid JSON: {str(e)}", "_raw": raw_args}
    
    def _fallback_parse(self, raw_args: str) -> Dict[str, Any]:
        """
        Simple fallback parser for common patterns of malformed JSON.
        """
        result = {}
        # Split by commas not in quotes
        in_quotes = False
        quote_char = None
        segments = []
        current = []
        
        for char in raw_args:
            if char in ['"', "'"]:
                if not in_quotes:
                    in_quotes = True
                    quote_char = char
                elif char == quote_char:
                    in_quotes = False
                    quote_char = None
            
            if char == ',' and not in_quotes:
                segments.append(''.join(current))
                current = []
            else:
                current.append(char)
        
        if current:
            segments.append(''.join(current))
            
        # Parse each key-value pair
        for segment in segments:
            if ':' in segment:
                key, value = segment.split(':', 1)
                key = key.strip().strip('"\'')
                value = value.strip()
                
                # Try to convert value to appropriate type
                if value.lower() == 'true':
                    result[key] = True
                elif value.lower() == 'false':
                    result[key] = False
                elif value.lower() == 'null':
                    result[key] = None
                elif value.startswith('"') and value.endswith('"'):
                    result[key] = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    result[key] = value[1:-1]
                else:
                    try:
                        result[key] = int(value)
                    except ValueError:
                        try:
                            result[key] = float(value)
                        except ValueError:
                            result[key] = value
        
        return result
