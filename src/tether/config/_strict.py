"""Shared base class for typed Settings sub-models.

Citation: _synthesis.md §13.4 M6 — StrictModel base. ``extra="forbid"`` catches
typos in YAML / dict input; ``frozen=True`` makes constructed sub-models
immutable so callers cannot mutate global config as a side effect.
"""
from pydantic import BaseModel, ConfigDict


class StrictModel(BaseModel):
    """Pydantic base for all Tether Settings sub-models.

    - ``extra="forbid"``: unknown fields raise ``ValidationError`` (typo guard).
    - ``frozen=True``: instances are immutable post-construction.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
