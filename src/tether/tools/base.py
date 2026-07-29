from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Literal, Optional, Type, get_args, get_origin

from pydantic import BaseModel, ConfigDict

from tether.core.interfaces import Tool

if TYPE_CHECKING:
    from tether.core.types import ToolExecutionContext


class ToolInputs(BaseModel):
    """Base class for **Style A** tool input models.

    Tools that opt into Pydantic-driven input validation declare a nested
    inputs model inheriting from :class:`ToolInputs` and assign it to the
    class-level :attr:`BaseTool.Inputs`. The :meth:`BaseTool.invoke` shim
    detects this and validates the args dict via
    :func:`Inputs.model_validate` before calling :meth:`run`.

    The default ``extra="forbid"`` matches the contract that the OpenAI
    function-call schema is the single source of truth: any model-emitted
    arg outside the schema is a model error, not silently dropped.

    Synthesis §4 Phase 4 step 43; A2 step 5.
    """

    model_config = ConfigDict(extra="forbid")

# =============================
# Tool Authoring Guidelines
# =============================
#
# To create a new tool:
# 1. Subclass BaseTool and implement the async run() method with explicit, type-annotated arguments.
# 2. Use a Google-style docstring for run() with an Args: section, e.g.:
#
#     async def run(self, timezone: str = "UTC", format: str = "human") -> dict:
#         """
#         Get the current time for a timezone in various formats.
#         Args:
#             timezone: IANA timezone (e.g., Europe/Dublin, America/New_York, UTC). Defaults to UTC if not provided.
#             format: The format for the returned time string. (e.g., "iso", "rfc2822", "human")
#         Returns:
#             dict: {"time": <formatted time string>}
#         """
#         ...
#
# 3. The schema will be generated automatically from the run() signature and docstring.
# 4. The class-level docstring will be used as the tool's description in the schema.
#
# Best Practices:
# - Always provide type hints for all arguments.
# - Use clear, concise parameter descriptions in the Args: section.
# - Document the return value in a Returns: section (optional, for clarity).
# - Avoid *args/**kwargs in run().
# - Use only JSON-serializable types for arguments and return values.

class BaseTool(Tool):

    REQUIRED: ClassVar[bool] = False
    """Whether the tool MUST be available for the engine to start.

    * ``REQUIRED = False`` (default): if :meth:`startup` raises, the tool is
      logged as a warning and dropped from the registry; the engine continues.
      This is the right policy for any tool whose absence merely degrades
      capability (e.g., ``WebSearchTool`` without ``BRAVE_API_KEY``).
    * ``REQUIRED = True``: a startup failure aborts engine startup
      (``Engine.__aenter__`` raises ``RuntimeError``). Reserved for tools
      whose absence makes the system meaningfully broken — none of the
      Phase 4 in-tree tools opt in. Future opt-in candidates: any
      authentication tool that the system prompt assumes is present.

    Synthesis §4 Phase 4 step 41 + §13.2 R5.
    """

    Inputs: ClassVar[Optional[Type[ToolInputs]]] = None
    """**Style A** opt-in: a :class:`ToolInputs` subclass that describes
    the tool's input schema declaratively.

    When set, :meth:`invoke` Pydantic-validates the args dict via
    ``Inputs.model_validate(args)`` and calls :meth:`run` with the
    validated model instance (single positional arg). When ``None``
    (the default — **Style B**), :meth:`invoke` unpacks the args dict
    as keyword arguments to :meth:`run`, the legacy behavior preserved
    for tools that prefer Annotated kwargs over a dedicated inputs class.

    Synthesis §4 Phase 4 step 43; A2 step 5.
    """

    def __init__(self):
        # No-op constructor preserved for subclasses that call ``super().__init__()``.
        # The legacy per-instance ``registry-name`` injection was retired in
        # Phase 4 step 43; the ``@tool(name=...)`` decorator now sets the
        # registry name at class definition time via the
        # ``_tether_tool_registered_name`` marker consumed by :attr:`name`.
        pass

    async def invoke(
        self,
        args: Dict[str, Any],
        *,
        context: Optional["ToolExecutionContext"] = None,
    ) -> Any:
        """Registry-facing shim. Three dispatch paths:

        1. **Style A** (``cls.Inputs`` is a :class:`ToolInputs` subclass):
           Pydantic-validate ``args`` via ``Inputs.model_validate(args)``
           and call ``run(inputs)`` (or ``run(inputs, context=...)`` if
           the signature opts in).
        2. **Style B with context**: ``run`` declares a keyword-only
           ``context`` param. Unpack args as kwargs and pass ``context``.
        3. **Style B legacy**: no ``context`` param. Unpack args as
           kwargs only.

        Connector-style tools (Phase 4.5+) opt into ``context`` to
        consume :attr:`ToolExecutionContext.user_confirmed_send` for
        the draft+confirm send-safety pattern (connector spec §4 footer).

        Synthesis §4 Phase 4 step 41a + step 43; A2 step 5.
        """
        import inspect

        if self.Inputs is not None:
            inputs = self.Inputs.model_validate(args)
            sig = inspect.signature(self.run)
            if "context" in sig.parameters:
                return await self.run(inputs, context=context)
            return await self.run(inputs)

        sig = inspect.signature(self.run)
        if "context" in sig.parameters:
            return await self.run(**args, context=context)
        return await self.run(**args)

    async def startup(self) -> None:
        """Open shared resources (httpx clients, DB connections, ...).

        Default no-op. Subclasses override when they own a resource that
        outlives a single ``invoke()`` call. Called once per
        ``Engine.__aenter__`` via
        :func:`tether.tools.lifecycle.startup_all`.

        If this method raises:

        * For a tool with ``REQUIRED = True``, ``startup_all`` re-raises
          (after the concurrent gather completes — synthesis §13.2 R5),
          which propagates out of ``Engine.__aenter__`` as
          ``RuntimeError``.
        * For ``REQUIRED = False`` (default), the failure is logged and
          the tool is dropped from the registry; the engine continues
          without it.

        Synthesis §4 Phase 4 step 41.
        """
        return None

    async def shutdown(self) -> None:
        """Close shared resources opened by :meth:`startup`.

        Default no-op. Called best-effort during ``Engine.aclose`` —
        before the hardware watchdog teardown so that tools can still
        depend on the provider during their own cleanup. Failures here
        are logged but never raised; partial cleanup is preferable to a
        crash that obscures the real shutdown reason.

        Phase 4.5 ordering caveat (rubber-duck consensus, xhigh
        OBSERVATION): when a tool is owned by a connector — i.e.
        returned by ``connector.tools()`` — :meth:`Engine.aclose` stops
        the connector FIRST (so no in-flight ``invoke()`` outlives its
        owner), then runs the tool's ``shutdown``. By the time this
        method runs for a connector-owned tool, the connector has
        already been stopped. Do NOT rely on connector resources here
        — fold any connector-dependent cleanup into the connector's
        own ``stop()`` method instead. The tool's ``shutdown`` should
        only release resources the tool itself owns (caches, files,
        in-memory state, etc.).

        Synthesis §4 Phase 4 step 41 + Phase 4.5 step 47d.
        """
        return None

    @staticmethod
    def _extract_param_descriptions(docstring: str) -> dict:
        """
        Parse the docstring for an Args: section and return a mapping of param name to description.
        Supports Google-style docstrings.
        """
        import re
        if not docstring:
            return {}
        param_desc = {}
        # Google style: Args:
        args_section = re.search(r"Args?:\s*(.*?)(^\S|\Z)", docstring, re.DOTALL | re.MULTILINE)
        if args_section:
            args_text = args_section.group(1)
            for line in args_text.splitlines():
                match = re.match(r"\s*(\w+)\s*:\s*(.*)", line)
                if match:
                    name, desc = match.groups()
                    param_desc[name] = desc.strip()
        return param_desc

    @staticmethod
    def _python_type_to_json_schema(t: Any, default: Any = ...) -> Dict[str, Any]:
        """Recursively convert a Python type annotation to a JSON Schema fragment.

        Synthesis §6 row 5 / R8 / connector spec §8.1:
        Single function — no class hierarchy (R6 anti-overengineering rule).

        Nullable convention: Optional[T] adds "nullable": true to the schema
        (OpenAPI 3.0 style). anyOf with null would be equally valid; we use the
        simpler nullable flag to keep emitted schemas compact.
        """
        import types as _types

        # Annotated[T, ...]: strip metadata, recurse on the inner type
        if get_origin(t) is _types.UnionType or get_origin(t) is None:
            pass  # handled below in the Union branch

        # typing.get_origin helpers for stdlib generics
        origin = get_origin(t)
        args = get_args(t)

        # Annotated[T, metadata, ...]  — ignore metadata, recurse on T
        try:
            import typing
            if origin is typing.Annotated:
                return BaseTool._python_type_to_json_schema(args[0], default)
        except AttributeError:
            pass

        # Optional[T]  ≡  Union[T, None]  (also handles T | None in 3.10+)
        is_union = (origin is getattr(__import__('typing'), 'Union', None))
        is_pep604_union = (
            hasattr(__import__('types'), 'UnionType') and
            isinstance(t, __import__('types').UnionType)
        )
        if is_union or is_pep604_union:
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner = BaseTool._python_type_to_json_schema(non_none[0])
                inner["nullable"] = True
                return inner
            # multi-type union without None: fall through to string fallback

        # list[T] / List[T]
        if origin is list:
            item_schema = (
                BaseTool._python_type_to_json_schema(args[0]) if args else {"type": "string"}
            )
            return {"type": "array", "items": item_schema}

        # dict[str, Any] / Dict[str, Any]
        if origin is dict:
            return {"type": "object"}

        # Literal["a", "b", ...]
        if origin is Literal:
            values = list(args)
            # Infer JSON type from first member
            first = values[0] if values else ""
            if isinstance(first, int):
                json_type = "integer"
            elif isinstance(first, float):
                json_type = "number"
            elif isinstance(first, bool):
                json_type = "boolean"
            else:
                json_type = "string"
            return {"type": json_type, "enum": values}

        # Primitive types
        _PRIMITIVES: Dict[Any, str] = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
        }
        if t in _PRIMITIVES:
            return {"type": _PRIMITIVES[t]}

        # Fallback: unknown type maps to string (preserves backward compat)
        return {"type": "string"}

    @property
    def auto_schema(self) -> Dict[str, Any]:
        """Generate an OpenAI-style function tool schema.

        Dispatch:

        * If :attr:`Inputs` is set (Style A), generate from
          ``cls.Inputs.model_json_schema()`` and strip Pydantic-specific
          noise so the result matches the convention used by the
          inspect-based path (no ``$defs``, no ``title`` keys, ``Optional``
          fields use ``nullable: true`` rather than ``anyOf [..., null]``).
        * Otherwise (Style B), introspect the :meth:`run` signature
          (with ``Annotated[T, Field(...)]`` metadata if present) and
          translate each parameter via
          :meth:`_python_type_to_json_schema`.

        Synthesis §4 Phase 4 step 43 + step 41 (auto_schema); A2 step 5/7.
        """
        inputs_cls = type(self).Inputs
        if inputs_cls is not None:
            params, required = self._schema_from_inputs(inputs_cls)
        else:
            params, required = self._schema_from_run_signature()
        return self.build_schema(
            function_name=self.name,
            description=self.__doc__ or "",
            parameters=params,
            required=required,
        )

    @staticmethod
    def _schema_from_inputs(
        inputs_cls: Type["ToolInputs"],
    ) -> "tuple[Dict[str, Any], List[str]]":
        """Extract ``(properties, required)`` from a :class:`ToolInputs` model.

        Strips Pydantic-specific noise so the emitted schema matches the
        Style B inspect-based convention:

        * Drops the top-level ``title`` and ``additionalProperties`` keys
          that Pydantic adds (``additionalProperties: false`` comes from
          ``extra="forbid"`` and is enforced at validate-time on the
          Python side, not in the wire schema).
        * Drops the per-property ``title`` keys.
        * Collapses ``anyOf: [{X}, {"type": "null"}]`` (Pydantic's
          encoding for ``Optional[X]``) into ``{X, "nullable": true}``
          to match the existing :meth:`_python_type_to_json_schema`
          convention.
        * Drops ``$defs`` if present (flat input models don't reference
          nested types; emitted defensively).

        Synthesis §4 Phase 4 step 43; A2 step 5.
        """
        raw = inputs_cls.model_json_schema()
        properties = raw.get("properties", {}) or {}
        required = list(raw.get("required", []) or [])

        cleaned: Dict[str, Any] = {}
        for name, prop in properties.items():
            cleaned[name] = BaseTool._clean_property_schema(prop)
        return cleaned, required

    @staticmethod
    def _clean_property_schema(prop: Dict[str, Any]) -> Dict[str, Any]:
        """Strip Pydantic noise from a single property schema fragment.

        Handles the ``Optional[T]`` case (``anyOf: [{T}, {"type": "null"}]``)
        by collapsing to a single fragment with ``nullable: true``.
        """
        prop = {k: v for k, v in prop.items() if k != "title"}

        any_of = prop.get("anyOf")
        if isinstance(any_of, list) and len(any_of) == 2:
            null_branch = next(
                (b for b in any_of if isinstance(b, dict) and b.get("type") == "null"),
                None,
            )
            other_branch = next(
                (b for b in any_of if not (isinstance(b, dict) and b.get("type") == "null")),
                None,
            )
            if null_branch is not None and isinstance(other_branch, dict):
                merged = {k: v for k, v in other_branch.items() if k != "title"}
                merged["nullable"] = True
                # Preserve sibling keys from the original (description, default).
                for k, v in prop.items():
                    if k in ("anyOf", "title"):
                        continue
                    merged.setdefault(k, v)
                return merged

        return prop

    def _schema_from_run_signature(
        self,
    ) -> "tuple[Dict[str, Any], List[str]]":
        """Style B path: introspect the typed :meth:`run` signature.

        Returns ``(properties, required)`` matching the contract that
        :meth:`auto_schema` then wraps with :meth:`build_schema`.

        Reads ``Annotated[T, Field(description=..., ge=..., ...)]``
        metadata to lift Pydantic ``Field`` constraints into the JSON
        schema (description, default, enum via ``Literal``, numeric
        bounds, etc.). Falls back to docstring ``Args:`` extraction
        when no Field metadata is present.

        Synthesis §4 Phase 4 step 43; A2 step 7.
        """
        import inspect
        from typing import get_type_hints

        from pydantic.fields import FieldInfo

        sig = inspect.signature(self.run)
        try:
            hints = get_type_hints(self.run, include_extras=True)
        except Exception:
            hints = {}
        docstring = self.run.__doc__ or self.__doc__ or ""
        param_docs = self._extract_param_descriptions(docstring)
        params: Dict[str, Any] = {}
        required: List[str] = []
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            if param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            if name == "context":
                # Phase 4 step 41a: ToolExecutionContext is plumbed by the
                # invoke shim, not exposed in the schema.
                continue

            t = hints.get(name, str)

            # Annotated[T, Field(...)]: lift Field metadata onto the schema.
            field_info: Optional[FieldInfo] = None
            origin = get_origin(t)
            try:
                import typing
                if origin is typing.Annotated:
                    inner_args = get_args(t)
                    inner_type = inner_args[0]
                    for meta in inner_args[1:]:
                        if isinstance(meta, FieldInfo):
                            field_info = meta
                            break
                    t = inner_type
            except AttributeError:
                pass

            prop = self._python_type_to_json_schema(t)

            if field_info is not None:
                if field_info.description:
                    prop["description"] = field_info.description
                # Numeric / string constraints expressed as Field metadata:
                for meta in field_info.metadata:
                    if hasattr(meta, "ge"):
                        prop["minimum"] = meta.ge
                    if hasattr(meta, "le"):
                        prop["maximum"] = meta.le
                    if hasattr(meta, "gt"):
                        prop["exclusiveMinimum"] = meta.gt
                    if hasattr(meta, "lt"):
                        prop["exclusiveMaximum"] = meta.lt
                    if hasattr(meta, "min_length"):
                        prop["minLength"] = meta.min_length
                    if hasattr(meta, "max_length"):
                        prop["maxLength"] = meta.max_length
                    if hasattr(meta, "pattern"):
                        prop["pattern"] = meta.pattern

            if "description" not in prop:
                prop["description"] = param_docs.get(name, "")

            if param.default is not inspect.Parameter.empty:
                prop["default"] = param.default
            else:
                required.append(name)
            params[name] = prop
        return params, required

    @property
    def name(self) -> str:
        """Registry name set by the :func:`@tool` decorator.

        Reads the class-level marker attribute
        (``_tether_tool_registered_name``) installed by
        :func:`tether.tools.registration.tool` at class
        definition time. Falls back to the bare class name when an
        undecorated :class:`BaseTool` subclass is used directly (test
        fixtures).

        Walks the MRO so the marker is found on whichever class in the
        chain was decorated, preferring the most-derived class's own
        marker. Synthesis §4 Phase 4 step 42 + step 43.
        """
        cls = type(self)
        for klass in cls.__mro__:
            if "_tether_tool_registered_name" in klass.__dict__:
                return klass.__dict__["_tether_tool_registered_name"]
        return cls.__name__


    @staticmethod
    def build_schema(
        function_name: str,
        description: str,
        parameters: dict,
        required: 'Optional[list[str]]' = None,
    ) -> dict:
        """
        Build a standard function tool schema.
        Args:
            function_name: Name of the function/tool.
            description: Description of the tool.
            parameters: Dict of parameter names to their JSON schema (type, description, etc).
            required: List of required parameter names.
        Returns:
            dict: Schema for the tool.
        """
        return {
            "type": "function",
            "function": {
                "name": function_name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required or [],
                },
            },
        }

    @property
    def schema(self) -> Dict[str, Any]:
        """
        Override in subclasses. Should return a standards-compliant tool schema dict.
        Use build_schema() for convenience.
        """
        return {}

    @abstractmethod
    async def run(self, *args, **kwargs) -> Any:
        """Author-facing API. Concrete tools override with typed signatures
        (e.g., `async def run(self, timezone: str = 'UTC')`).

        NOT part of the Tool ABC — do not call from the orchestrator or
        ToolRunner. Use tool.invoke(args) instead (synthesis §6 row 4).
        """
        raise NotImplementedError()
