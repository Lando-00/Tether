import re
from typing import List, Optional, Tuple, Dict, Any
from llm_service.protocol.core.interfaces import ToolCallStrategy, Logger, ConfigProvider
from llm_service.protocol.core.loggers import NoOpLogger
from llm_service.protocol.core.config import EnvironmentConfigProvider

class PrefixedToolCallDetector(ToolCallStrategy):
    """
    Detects __tool_<n>(...) via early tokens with balanced parens.
    """
    CALL_PREFIX_RE = re.compile(r'^__tool_[A-Za-z_]\w*\([^)]*$')
    CALL_FULL_RE   = re.compile(r'^__tool_[A-Za-z_]\w*\([^)]*\)\s*$')
    # Matches a complete tool call anywhere in the token sequence
    ANYWHERE_FULL  = re.compile(r'__tool_[A-Za-z_]\w*\([^)]*\)\s*')
    # Matches a partial tool call (no closing paren) anywhere
    ANYWHERE_PREFX = re.compile(r'__tool_[A-Za-z_]\w*\([^)]*$')

    def __init__(self, prefix: str = "__tool_", logger: Optional[Logger] = None, config: Optional[ConfigProvider] = None):
        self.prefix = prefix
        self._raw: List[str] = []
        self._compact: List[str] = []
        self._decided_prose = False
        self._started = False
        self._name: Optional[str] = None
        self._paren_balance = 0
        self._first_paren_ix: Optional[int] = None
        self._logger = logger or NoOpLogger()
        self._config = config or EnvironmentConfigProvider()
        self._max_buffer_size = self._config.get_max_token_buffer_size()

    def feed(self, token: str) -> Tuple[str, Optional[str]]:
        try:
            # Check buffer limit to prevent memory leaks
            if len(self._raw) >= self._max_buffer_size:
                self._logger.warning("Token buffer size exceeded maximum limit of %d", self._max_buffer_size)
                self._decided_prose = True
                return ("prose", "".join(self._raw[:100]) + "... [buffer overflow]")
            
            self._raw.append(token)
            if token and token.strip():
                self._compact.append(token.replace(" ", ""))

            raw = "".join(self._raw)
            compact = "".join(self._compact)

            if self._decided_prose:
                return ("undecided", None)

            if not self._started:
                # Relax early rejection: ignore leading whitespace, reject only if many newlines
                raw_stripped = raw.lstrip()
                newline_count = raw_stripped.count("\n")
                too_many_newlines = newline_count > 4
                too_long = len(raw) > 512 or len(compact) > 256
                # Detect tool call anywhere in prefix for complete or partial matches
                m_full = self.ANYWHERE_FULL.search(compact)
                if m_full:
                    self._started = True
                    self._name = self._extract_name(m_full.group(0))
                    self._logger.info("==== DETECTED TOOL CALL: %s ====", self._name)
                    self._logger.debug("Found complete tool call pattern: %s", m_full.group(0))
                    return ("call_started", self._name)
                m_pre = self.ANYWHERE_PREFX.search(compact)
                if m_pre:
                    self._started = True
                    self._name = self._extract_name(m_pre.group(0))
                    self._first_paren_ix = compact.find("(")
                    self._paren_balance = 1
                    self._logger.debug("Started tool call detection anywhere: %s", self._name)
                    return ("call_started", self._name)
                # Standard prefix/full-match detection
                looks_full = bool(self.CALL_FULL_RE.match(compact))
                looks_prefix = bool(self.CALL_PREFIX_RE.match(compact))
                if looks_full:
                    self._started = True
                    self._name = self._extract_name(compact)
                    self._logger.debug("Detected complete tool call: %s", self._name)
                    return ("call_started", self._name)
                if looks_prefix:
                    self._started = True
                    self._name = self._extract_name(compact)
                    self._first_paren_ix = compact.find("(")
                    self._paren_balance = 1
                    self._logger.debug("Started tool call detection: %s", self._name)
                    return ("call_started", self._name)
                # Immediate rejections: too many newlines or excessive length
                if too_many_newlines or too_long:
                    self._decided_prose = True
                    self._logger.debug("Rejecting as prose (too many newlines: %d or too long)", newline_count)
                    return ("prose", raw)
                
                # Be lenient with partial prefixes - check if what we have so far could
                # be the beginning of the tool prefix
                if len(compact) <= len(self.prefix) and self.prefix.startswith(compact):
                    self._logger.debug("Potential tool prefix start detected: %s", compact)
                    return ("undecided", None)
                
                # If we've collected enough and it doesn't match our prefix pattern
                if not compact.startswith(self.prefix):
                    self._decided_prose = True
                    self._logger.debug("Rejecting as prose: doesn't match tool prefix pattern")
                    return ("prose", raw)

                return ("undecided", None)

            # After start: update paren balance
            for ch in token:
                if ch == "(":
                    self._paren_balance += 1
                elif ch == ")":
                    self._paren_balance -= 1

            if self._paren_balance <= 0:
                compact = "".join(self._compact)
                if self._first_paren_ix is None:
                    self._first_paren_ix = compact.find("(")
                last_close = compact.rfind(")")
                
                # Debug log about parenthesis finding
                self._logger.debug("Parenthesis locations: first_open=%s, last_close=%s",
                               self._first_paren_ix, last_close)
                
                inner = compact[self._first_paren_ix + 1:last_close] if last_close != -1 else ""
                self._logger.info("==== COMPLETED TOOL CALL: %s ====", self._name)
                self._logger.info("Tool call arguments (raw): %s", inner)
                
                # Print full raw buffer for debugging
                raw_text = "".join(self._raw)
                self._logger.debug("Complete raw input: %s", raw_text[:200] + ("..." if len(raw_text) > 200 else ""))
                return ("call_complete", inner)

            return ("undecided", None)
        except Exception as e:
            self._logger.exception("Error in tool call detection: %s", str(e))
            self._decided_prose = True
            return ("prose", "".join(self._raw[:100]) + f"... [error: {str(e)}]")

    def _extract_name(self, compact: str) -> str:
        open_ix = compact.find("(")
        return compact[len(self.prefix):open_ix] if open_ix != -1 else ""
