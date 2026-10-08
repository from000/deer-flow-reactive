"""Tool contracts consumed by the agent layer.

This module owns the observable surface of the tool primitives DeerFlow uses:
the ``@tool`` decorator (bare, named, and keyword forms), ``StructuredTool``,
the generated pydantic ``args_schema`` / ``tool_call_schema`` pair, injected
argument markers, and the OpenAI function/tool schema conversion helpers.

It deliberately imports no external agent framework: tools are plain pydantic
models plus callables, which lets the engine replace the runtime underneath a
real agent factory without changing tool definitions or model-facing schemas.

Portions of this module are adapted from upstream `langchain-core`
(https://github.com/langchain-ai/langchain) so that hosts written
against the LangChain / LangGraph surface keep working on the
ReactiveGraph engine. See THIRD_PARTY_NOTICES.md for the upstream
MIT copyright notices.
"""

from __future__ import annotations

import functools
import inspect
import json
import textwrap
import typing
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Annotated, Any, Literal, cast, get_args, get_origin, get_type_hints

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, ValidationError, create_model
from pydantic.fields import FieldInfo
from typing_extensions import is_typeddict

from reactivegraph.messages import ToolMessage, ToolOutputMixin, is_tool_output
from reactivegraph.middleware import (
    HostDirectlyInjectedToolArg,
    ToolRuntime,
    _DirectlyInjectedToolArg,
)

__all__ = (
    "FILTERED_ARGS",
    "ArgsSchema",
    "BaseTool",
    "InjectedToolArg",
    "InjectedToolCallId",
    "SchemaAnnotationError",
    "StructuredTool",
    "ToolException",
    "ToolRuntime",
    "convert_to_openai_function",
    "convert_to_openai_tool",
    "create_schema_from_function",
    "tool",
)

FILTERED_ARGS = ("run_manager", "callbacks")
TOOL_MESSAGE_BLOCK_TYPES = (
    "text",
    "image_url",
    "image",
    "json",
    "search_result",
    "custom_tool_call_output",
    "document",
    "file",
)

_EMPTY_SET: frozenset[str] = frozenset()


class SchemaAnnotationError(TypeError):
    """Raised when ``args_schema`` is missing or has an incorrect annotation."""


class ToolException(Exception):  # noqa: N818
    """Exception thrown when a tool execution error occurs."""


class InjectedToolArg:
    """Annotation for tool arguments injected at runtime.

    Tool arguments annotated with this marker are excluded from the schema sent
    to language models and supplied by the execution runtime instead.
    """


class InjectedToolCallId(InjectedToolArg):
    """Annotation for injecting the tool call ID into a tool argument."""


ArgsSchema = type[BaseModel] | dict[str, Any]


def _is_annotated_type(typ: Any) -> bool:
    return get_origin(typ) in {typing.Annotated, Annotated}


def _get_annotation_description(arg_type: Any) -> str | None:
    """Extract a description from an ``Annotated`` type."""
    if _is_annotated_type(arg_type):
        for annotation in get_args(arg_type)[1:]:
            if isinstance(annotation, str):
                return annotation
            if isinstance(annotation, FieldInfo) and annotation.description:
                return annotation.description
    return None


_DOCSTRING_BREAKERS = ("Returns:", "Example:")
_DOCSTRING_IGNORED_ARGS = {"run_manager", "callbacks", "runtime", "return"}


def _split_google_docstring(
    docstring: str, args: list[str], *, error_on_invalid_docstring: bool
) -> tuple[str, str | None]:
    """Return ``(description, args_block)`` from a Google-style docstring."""
    blocks = docstring.split("\n\n")
    if error_on_invalid_docstring:
        filtered = {arg for arg in args if arg not in _DOCSTRING_IGNORED_ARGS}
        if filtered and (
            len(blocks) < 2
            or not any(block.startswith("Args:") for block in blocks[1:])
        ):
            raise ValueError("Found invalid Google-Style docstring.")
    descriptors: list[str] = []
    args_block: str | None = None
    past_descriptors = False
    for block in blocks:
        if block.startswith("Args:"):
            args_block = block
            break
        if block.startswith(_DOCSTRING_BREAKERS):
            past_descriptors = True
        elif not past_descriptors:
            descriptors.append(block)
    return " ".join(descriptors).strip(), args_block


def _parse_google_arg_block(args_block: str) -> dict[str, str]:
    """Parse the ``Args:`` section into ``{arg_name: description}``."""
    arg_descriptions: dict[str, str] = {}
    arg: str | None = None
    arg_indent: int | None = None
    for line in args_block.split("\n")[1:]:
        if not line.strip():
            continue
        current_indent = len(line) - len(line.lstrip())
        if arg_indent is None and ":" in line:
            arg_indent = current_indent
        is_continuation = arg_indent is not None and current_indent > arg_indent
        if arg is not None and is_continuation:
            arg_descriptions[arg] += " " + line.strip()
        elif ":" in line:
            arg, desc = line.split(":", maxsplit=1)
            arg = arg.strip()
            arg_name, _, annotations = arg.partition(" ")
            if annotations.startswith("(") and annotations.endswith(")"):
                arg = arg_name
            arg_descriptions[arg] = desc.strip()
        elif arg:
            arg_descriptions[arg] += " " + line.strip()
    return arg_descriptions


def _parse_google_docstring(
    docstring: str | None,
    args: list[str],
    *,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict[str, str]]:
    """Parse a Google-style docstring into description + argument descriptions."""
    if not docstring:
        if error_on_invalid_docstring:
            raise ValueError("Found invalid Google-Style docstring.")
        return "", {}
    description, args_block = _split_google_docstring(
        docstring, args, error_on_invalid_docstring=error_on_invalid_docstring
    )
    return description, _parse_google_arg_block(args_block) if args_block else {}


def _infer_arg_descriptions(
    fn: Callable[..., Any],
    *,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
) -> tuple[str, dict[str, str]]:
    """Infer tool + argument descriptions from a docstring and annotations."""
    annotations = _resolve_type_hints(fn)
    if parse_docstring:
        description, arg_descriptions = _parse_google_docstring(
            inspect.getdoc(fn),
            list(annotations),
            error_on_invalid_docstring=error_on_invalid_docstring,
        )
    else:
        description = inspect.getdoc(fn) or ""
        arg_descriptions = {}

    if parse_docstring:
        for docstring_arg in arg_descriptions:
            if docstring_arg not in annotations:
                msg = f"Arg {docstring_arg} in docstring not found in function signature."
                raise ValueError(msg)
    for arg, arg_type in annotations.items():
        if arg in arg_descriptions:
            continue
        if desc := _get_annotation_description(arg_type):
            arg_descriptions[arg] = desc
    return description, arg_descriptions


def _is_directly_injected_arg_type(type_: Any) -> bool:
    """Check for directly-injected argument types such as ``ToolRuntime``.

    Two markers must be accepted. ``_DirectlyInjectedToolArg`` is the engine's
    native bridge for annotations written against it, while a bridged
    ``ToolRuntime`` inherits the *host* class and therefore the host marker
    (the two are different class objects). Checking only one of them drops the
    runtime back into the generated schema, whose ``Callable`` field then dies
    in JSON-schema generation.
    """
    markers: tuple[type, ...] = (
        (_DirectlyInjectedToolArg, HostDirectlyInjectedToolArg)
        if HostDirectlyInjectedToolArg is not None
        else (_DirectlyInjectedToolArg,)
    )
    if isinstance(type_, type) and issubclass(type_, markers):
        return True
    origin = get_origin(type_)
    return (
        isinstance(origin, type)
        and issubclass(origin, markers)
    )


def _is_injected_arg_type(
    type_: Any, injected_type: type[InjectedToolArg] | None = None
) -> bool:
    """Check whether an annotation marks an injected tool argument."""
    if injected_type is None:
        if _is_directly_injected_arg_type(type_):
            return True
        injected_type = InjectedToolArg
    return any(
        isinstance(arg, injected_type)
        or (isinstance(arg, type) and issubclass(arg, injected_type))
        for arg in get_args(type_)[1:]
    )


def _create_subset_model(
    name: str,
    model: type[BaseModel],
    field_names: list[str],
    *,
    descriptions: dict[str, str] | None = None,
    fn_description: str | None = None,
) -> type[BaseModel]:
    """Create a pydantic model containing only *field_names* of *model*."""
    descriptions_ = descriptions or {}
    fields: dict[str, Any] = {}
    for field_name in field_names:
        field = model.model_fields[field_name]
        description = descriptions_.get(field_name, field.description)
        field_kwargs: dict[str, Any] = {"description": description}
        if field.default_factory is not None:
            field_kwargs["default_factory"] = field.default_factory
        else:
            field_kwargs["default"] = field.default
        field_info = FieldInfo(**field_kwargs)
        if field.metadata:
            field_info.metadata = field.metadata
        fields[field_name] = (field.annotation, field_info)

    rtn = create_model(
        name,
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )
    selected_annotations = [
        (field_name, annotation)
        for field_name, annotation in model.__annotations__.items()
        if field_name in field_names
    ]
    rtn.__annotations__ = dict(selected_annotations)
    rtn.__doc__ = textwrap.dedent(fn_description or model.__doc__ or "")
    return rtn


def _resolve_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    """Resolve annotations with enclosing-frame locals.

    Functions created inside another function carry string annotations under
    ``from __future__ import annotations``; the names they reference (locally
    defined models, locally imported markers) are only present in the defining
    frame.  Walk the stack outermost-first so the innermost binding wins, which
    matches how pydantic's validation decorator resolves parent namespaces.
    """
    frames = []
    frame = inspect.currentframe()
    try:
        while frame is not None:
            frames.append(frame)
            frame = frame.f_back
        localns: dict[str, Any] = {}
        for enclosing in reversed(frames):
            localns.update(enclosing.f_locals)
        return get_type_hints(
            func,
            globalns=getattr(func, "__globals__", None),
            localns=localns,
            include_extras=True,
        )
    finally:
        del frames
        del frame


def _get_type_hints(func: Callable[..., Any]) -> dict[str, Any]:
    return _resolve_type_hints(func)


def _get_type_hints_or_empty(func: Callable[..., Any]) -> dict[str, Any]:
    try:
        return _resolve_type_hints(func)
    except Exception:  # noqa: BLE001
        return {}


def _get_all_basemodel_annotations(model: type[BaseModel]) -> dict[str, Any]:
    """Return field annotations with ``Annotated`` metadata preserved.

    ``model_fields[name].annotation`` strips ``Annotated`` metadata into
    ``FieldInfo.metadata``, so injected-argument detection must read the model's
    own annotations instead.
    """
    try:
        return get_type_hints(model, include_extras=True)
    except Exception:  # noqa: BLE001
        return dict(getattr(model, "__annotations__", {}))


def _get_runnable_config_param(func: Callable[..., Any]) -> str | None:
    """Return the parameter name annotated as a runnable config, if any."""
    for name, type_ in _get_type_hints_or_empty(func).items():
        if getattr(type_, "__name__", None) == "RunnableConfig":
            return name
    return None


def create_schema_from_function(
    model_name: str,
    func: Callable[..., Any],
    *,
    filter_args: Sequence[str] | None = None,
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = False,
    include_injected: bool = True,
) -> type[BaseModel]:
    """Create a pydantic schema from a function's signature."""
    sig = inspect.signature(func)
    in_class = bool(func.__qualname__ and "." in func.__qualname__)
    has_args = any(
        param.kind == param.VAR_POSITIONAL for param in sig.parameters.values()
    )
    has_kwargs = any(
        param.kind == param.VAR_KEYWORD for param in sig.parameters.values()
    )

    resolved_hints = _resolve_type_hints(func)
    annotations: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name in {"self", "cls"} and in_class:
            continue
        if param.kind in {param.VAR_POSITIONAL, param.VAR_KEYWORD}:
            continue
        annotation = resolved_hints.get(
            name, Any if param.annotation is param.empty else param.annotation
        )
        annotations[name] = annotation

    if filter_args:
        filter_args_ = list(filter_args)
    else:
        filter_args_ = list(FILTERED_ARGS)
        if not include_injected:
            for existing_param, annotation in annotations.items():
                if _is_injected_arg_type(annotation):
                    filter_args_.append(existing_param)

    description, arg_descriptions = _infer_arg_descriptions(
        func,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
    )

    field_definitions: dict[str, Any] = {}
    for name, param in sig.parameters.items():
        if name not in annotations:
            continue
        if name in filter_args_:
            continue
        default: Any = ... if param.default is param.empty else param.default
        field_definitions[name] = (annotations[name], default)

    if has_args:
        field_definitions.setdefault("args", (tuple[Any, ...], ()))
    if has_kwargs:
        field_definitions.setdefault("kwargs", (dict[str, Any], {}))

    model = create_model(
        model_name,
        __config__=ConfigDict(extra="forbid", arbitrary_types_allowed=True),
        **field_definitions,
    )
    return _create_subset_model(
        model_name,
        model,
        list(field_definitions),
        descriptions=arg_descriptions,
        fn_description=description,
    )


def _recursive_set_additional_properties_false(
    schema: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(schema, dict):
        if "required" in schema or not schema:
            schema = {**schema, "additionalProperties": False}
        if "properties" in schema:
            schema = {
                **schema,
                "properties": {
                    name: _recursive_set_additional_properties_false(value)
                    for name, value in schema["properties"].items()
                },
            }
        if "$defs" in schema:
            schema = {
                **schema,
                "$defs": {
                    name: _recursive_set_additional_properties_false(value)
                    for name, value in schema["$defs"].items()
                },
            }
    return schema


def _dereference_refs(
    obj: Any, *, full_schema: dict[str, Any] | None = None, skip_keys: Sequence[str] | None = None
) -> Any:
    """Inline local ``$ref`` entries so the produced schema is self-contained."""
    full_schema = full_schema or obj
    skip_keys = skip_keys or []
    if not isinstance(obj, dict):
        if isinstance(obj, list):
            return [
                _dereference_refs(item, full_schema=full_schema, skip_keys=skip_keys)
                for item in obj
            ]
        return obj
    result: dict[str, Any] = {}
    for key, value in obj.items():
        if key in skip_keys:
            result[key] = value
            continue
        if key == "$ref":
            ref = value.rsplit("/", 1)[-1]
            ref_value = full_schema.get("$defs", {}).get(ref) or full_schema.get(
                "definitions", {}
            ).get(ref)
            if ref_value is not None:
                dereferenced = _dereference_refs(
                    ref_value, full_schema=full_schema, skip_keys=skip_keys
                )
                for sub_key, sub_value in dereferenced.items():
                    result.setdefault(sub_key, sub_value)
                continue
        result[key] = _dereference_refs(
            value, full_schema=full_schema, skip_keys=skip_keys
        )
    return result


def _rm_titles(kv: Any, prev_key: str = "") -> Any:
    if isinstance(kv, dict):
        return {
            key: _rm_titles(value, key)
            for key, value in kv.items()
            if key != "title" or prev_key in {"anyOf", "oneOf", "allOf", "items"}
        }
    if isinstance(kv, list):
        return [_rm_titles(item) for item in kv]
    return kv


def _convert_json_schema_to_openai_function(
    schema: dict[str, Any],
    *,
    name: str | None = None,
    description: str | None = None,
    rm_titles: bool = True,
) -> dict[str, Any]:
    schema = _dereference_refs(schema)
    schema.pop("definitions", None)
    schema.pop("$defs", None)
    title = schema.pop("title", "")
    default_description = schema.pop("description", "")
    return {
        "name": name or title,
        "description": description or default_description,
        "parameters": _rm_titles(schema) if rm_titles else schema,
    }


def _convert_pydantic_to_openai_function(
    model: type[BaseModel],
    *,
    name: str | None = None,
    description: str | None = None,
    rm_titles: bool = True,
) -> dict[str, Any]:
    schema = model.model_json_schema()
    return _convert_json_schema_to_openai_function(
        schema, name=name, description=description, rm_titles=rm_titles
    )


_OPENAI_FUNCTION_KEYS = {"name", "description", "parameters", "strict"}


def _dict_to_openai_function(function: dict[str, Any]) -> dict[str, Any]:
    """Convert the supported dict shorthands to an OpenAI function."""
    if all(key in function for key in ("name", "input_schema")):
        oai_function: dict[str, Any] = {
            "name": function["name"],
            "parameters": function["input_schema"],
        }
        if "description" in function:
            oai_function["description"] = function["description"]
        return oai_function
    if "toolSpec" in function:
        spec = function["toolSpec"]
        oai_function = {
            "name": spec["name"],
            "parameters": spec["inputSchema"]["json"],
        }
        if "description" in spec:
            oai_function["description"] = spec["description"]
        return oai_function
    if "name" in function:
        return {
            key: value
            for key, value in function.items()
            if key in _OPENAI_FUNCTION_KEYS
        }
    if "title" in function:
        remainder = function.copy()
        oai_function = {"name": remainder.pop("title")}
        if "description" in remainder:
            oai_function["description"] = remainder.pop("description")
        if remainder and "properties" in remainder:
            oai_function["parameters"] = remainder
        return oai_function
    raise ValueError(
        f"Unsupported function\n\n{function}\n\nTo use a JSON schema as a "
        "function, it must have a top-level 'title' key to be used as the "
        "function name."
    )


def convert_to_openai_function(
    function: Any, *, strict: bool | None = None
) -> dict[str, Any]:
    """Convert a tool-like object to the inner OpenAI function description.

    Accepts an OpenAI/Anthropic dict shorthand, a pydantic model, a
    ``TypedDict``, a :class:`BaseTool`, or a plain callable.
    """
    if isinstance(function, dict):
        oai_function = _dict_to_openai_function(function)
    elif isinstance(function, type) and issubclass(function, BaseModel):
        oai_function = _convert_pydantic_to_openai_function(function)
    elif is_typeddict(function):
        schema = create_model(
            "schema",
            **{k: (v, ...) for k, v in function.__annotations__.items()},
        )
        oai_function = _convert_pydantic_to_openai_function(schema)
    elif isinstance(function, BaseTool):
        oai_function = _format_tool_to_openai_function(function)
    elif callable(function):
        model = create_schema_from_function(
            function.__name__,
            function,
            filter_args=(),
            parse_docstring=True,
            error_on_invalid_docstring=False,
            include_injected=False,
        )
        oai_function = _convert_pydantic_to_openai_function(
            model, name=function.__name__, description=model.__doc__
        )
    else:
        raise ValueError(
            f"Unsupported function\n\n{function}\n\nFunctions must be passed in as "
            "Dict, pydantic.BaseModel, or Callable. If they're a dict they must either "
            "be in OpenAI function format or valid JSON schema with top-level 'title' key."
        )

    if strict is not None:
        if "strict" in oai_function and oai_function["strict"] != strict:
            msg = (
                f"Tool/function already has a 'strict' key with value "
                f"{oai_function['strict']} which is different from the explicit "
                f"`strict` arg received {strict=}."
            )
            raise ValueError(msg)
        oai_function["strict"] = strict
        if strict:
            parameters = oai_function.get("parameters")
            if isinstance(parameters, dict):
                fields = parameters.get("properties")
                if isinstance(fields, dict) and fields:
                    parameters = dict(parameters)
                    parameters["required"] = list(fields.keys())
                    oai_function["parameters"] = parameters
            oai_function["parameters"] = _recursive_set_additional_properties_false(
                oai_function["parameters"]
            )
    return oai_function


_WELL_KNOWN_OPENAI_TOOLS = (
    "function",
    "file_search",
    "computer",
    "computer_use_preview",
    "code_interpreter",
    "mcp",
    "image_generation",
    "web_search_preview",
    "web_search",
    "tool_search",
    "apply_patch",
    "namespace",
)


def convert_to_openai_tool(
    tool: Mapping[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool,
    *,
    strict: bool | None = None,
) -> dict[str, Any]:
    """Convert a tool-like object to an OpenAI tool schema."""
    if isinstance(tool, dict):
        if tool.get("type") in _WELL_KNOWN_OPENAI_TOOLS:
            return dict(tool)
        if (tool.get("type") or "").startswith("web_search_preview"):
            return dict(tool)
    oai_function = convert_to_openai_function(tool, strict=strict)
    return {"type": "function", "function": oai_function}


def _format_tool_to_openai_function(tool: BaseTool) -> dict[str, Any]:
    """Format a tool into the OpenAI function API."""
    schema = tool.tool_call_schema
    if isinstance(schema, dict):
        return _convert_json_schema_to_openai_function(
            schema, name=tool.name, description=tool.description
        )
    return _convert_pydantic_to_openai_function(
        schema, name=tool.name, description=tool.description
    )


def _host_base_tool() -> type | None:
    """Return langchain-core's ``BaseTool`` as an extra base, if installed.

    Hosts hand our tools to *their* ``ToolNode`` (``langchain.agents``
    ``create_agent`` collects ``middleware.tools`` and builds one). ``ToolNode``
    does ``isinstance(tool, langchain_core.tools.BaseTool)`` and, when that is
    false, tries to re-coerce the object through ``create_tool`` — which dies
    with ``ValueError: The first argument must be a string or a callable…``.
    A structurally identical but unrelated class therefore cannot cross the
    boundary; inheriting the real host class is the only way to be a drop-in
    base. Measured on the real DeerFlow suite: every TodoMiddleware graph test
    failed this way.

    langchain-core stays optional: without it this returns ``None`` and the
    classes are plain pydantic models with the identical surface.
    """
    try:
        from langchain_core.tools import BaseTool as HostBaseTool
    except ImportError:  # pragma: no cover - langchain-core is an optional host dep
        return None
    return HostBaseTool


HostBaseTool = _host_base_tool()
_ToolBase = HostBaseTool or BaseModel


class BaseTool(
    _ToolBase,  # type: ignore[misc, valid-type]
):
    """Base class for all tools."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    args_schema: Annotated[ArgsSchema | None, Field(default=None, exclude=True)]
    return_direct: bool = False
    verbose: bool = False
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None
    handle_tool_error: bool | str | Callable[[ToolException], Any] | None = False
    handle_validation_error: bool | str | Callable[[Exception], str] | None = False
    response_format: Literal["content", "content_and_artifact"] = "content"

    _tool_call_schema_memo: ArgsSchema | None = PrivateAttr(default=None)

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        args_schema_type = cls.__annotations__.get("args_schema", None)
        if args_schema_type is not None and args_schema_type == BaseModel:
            typehint_mandate = """
class ChildTool(BaseTool):
    ...
    args_schema: Type[BaseModel] = SchemaClass
    ..."""
            msg = (
                f"Tool definition for {cls.__name__} must include valid type annotations"
                f" for argument 'args_schema' to behave as expected.\n"
                f"Expected annotation of 'Type[BaseModel]'"
                f" but got '{args_schema_type}'.\n"
                f"Expected class looks like:\n"
                f"{typehint_mandate}"
            )
            raise SchemaAnnotationError(msg)

    @property
    def is_single_input(self) -> bool:
        """Whether the tool accepts only a single input argument."""
        return len({key for key in self.args if key != "kwargs"}) == 1

    @property
    def args(self) -> dict[str, Any]:
        """The tool's input-argument schema properties."""
        schema = self.tool_call_schema
        if isinstance(schema, dict):
            json_schema = schema
        else:
            json_schema = schema.model_json_schema()
        return cast("dict[str, Any]", json_schema["properties"])

    @property
    def tool_call_schema(self) -> ArgsSchema:
        """The schema exposed to language models, excluding injected arguments."""
        if isinstance(self.args_schema, dict):
            if self.description:
                return {**self.args_schema, "description": self.description}
            return self.args_schema
        if (memo := self._tool_call_schema_memo) is not None:
            return memo
        full_schema = self.get_input_schema()
        fields = [
            field_name
            for field_name, type_ in _get_all_basemodel_annotations(full_schema).items()
            if not _is_injected_arg_type(type_)
        ]
        subset_model = _create_subset_model(
            self.name, full_schema, fields, fn_description=self.description
        )
        self._tool_call_schema_memo = subset_model
        return subset_model

    @property
    def _injected_args_keys(self) -> frozenset[str]:
        return _EMPTY_SET

    def get_input_schema(self) -> type[BaseModel]:
        """The tool's full input schema (including injected arguments)."""
        if self.args_schema is not None:
            if isinstance(self.args_schema, dict):
                return create_model(
                    self.name,
                    __config__=ConfigDict(arbitrary_types_allowed=True),
                )
            return self.args_schema
        return create_schema_from_function(self.name, self._run)

    def _parse_input(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> str | dict[str, Any]:
        input_args = self.args_schema
        if isinstance(tool_input, str):
            if input_args is not None:
                if isinstance(input_args, dict):
                    msg = (
                        "String tool inputs are not allowed when "
                        "using tools with JSON schema args_schema."
                    )
                    raise ValueError(msg)
                key_ = next(iter(input_args.model_fields))
                input_args.model_validate({key_: tool_input})
            return tool_input

        if input_args is not None:
            if isinstance(input_args, dict):
                return tool_input
            for key, value in _get_all_basemodel_annotations(input_args).items():
                if _is_injected_arg_type(value, injected_type=InjectedToolCallId):
                    if tool_call_id is None:
                        msg = (
                            "When tool includes an InjectedToolCallId "
                            "argument, tool must always be invoked with a full "
                            "model ToolCall of the form: {'args': {...}, "
                            "'name': '...', 'type': 'tool_call', "
                            "'tool_call_id': '...'}"
                        )
                        raise ValueError(msg)
                    tool_input[key] = tool_call_id
            result = input_args.model_validate(tool_input)
            result_dict = result.model_dump()
            validated_input: dict[str, Any] = {}
            for key in result_dict:
                if key in tool_input:
                    validated_input[key] = getattr(result, key)
                elif key in input_args.model_fields and key not in {"args", "kwargs"}:
                    field = input_args.model_fields[key]
                    if not field.is_required():
                        validated_input[key] = getattr(result, key)
            for key in self._injected_args_keys:
                if key in tool_input:
                    validated_input[key] = tool_input[key]
                elif key == "tool_call_id":
                    if tool_call_id is None:
                        msg = (
                            "When tool includes an InjectedToolCallId "
                            "argument, tool must always be invoked with a full "
                            "model ToolCall of the form: {'args': {...}, "
                            "'name': '...', 'type': 'tool_call', "
                            "'tool_call_id': '...'}"
                        )
                        raise ValueError(msg)
                    validated_input[key] = tool_call_id
            return validated_input
        return tool_input

    def _filter_injected_args(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        filtered_keys = set(FILTERED_ARGS)
        filtered_keys.update(self._injected_args_keys)
        if self.args_schema is not None and not isinstance(self.args_schema, dict):
            for field_name, type_ in _get_all_basemodel_annotations(
                self.args_schema
            ).items():
                if _is_injected_arg_type(type_):
                    filtered_keys.add(field_name)
        return {key: value for key, value in tool_input.items() if key not in filtered_keys}

    def _to_args_and_kwargs(
        self, tool_input: str | dict[str, Any], tool_call_id: str | None
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        if (
            self.args_schema is not None
            and isinstance(self.args_schema, type)
            and issubclass(self.args_schema, BaseModel)
            and not self.args_schema.model_fields
        ):
            return (), {}
        tool_input = self._parse_input(tool_input, tool_call_id)
        if isinstance(tool_input, str):
            return (tool_input,), {}
        if isinstance(tool_input, dict):
            return (), tool_input.copy()
        msg = f"Invalid tool input type: {type(tool_input)}"
        raise TypeError(msg)

    def _handle_error(self, error: ToolException) -> Any:
        flag = self.handle_tool_error
        if flag is True:
            return str(error)
        if isinstance(flag, str):
            return flag
        if callable(flag):
            return flag(error)
        raise error

    def _handle_validation_error(self, error: Exception) -> Any:
        flag = self.handle_validation_error
        if flag is True:
            return str(error)
        if isinstance(flag, str):
            return flag
        if callable(flag):
            return flag(error)
        raise error

    def run(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,
        *,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool synchronously."""
        content: Any = None
        artifact: Any = None
        status: Literal["success", "error"] = "success"
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            response = self._run(*tool_args, **tool_kwargs)
        except ValidationError as exc:
            if not self.handle_validation_error:
                raise
            content = self._handle_validation_error(exc)
            status = "error"
        except ToolException as exc:
            if not self.handle_tool_error:
                raise
            content = self._handle_error(exc)
            status = "error"
        else:
            if self.response_format == "content_and_artifact":
                if not isinstance(response, tuple) or len(response) != 2:
                    msg = (
                        "Since response_format='content_and_artifact' "
                        "a two-tuple of the message content and raw tool output is "
                        f"expected. Instead, generated response is of type: "
                        f"{type(response)}."
                    )
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
        return _format_output(content, artifact, tool_call_id, self.name, status)

    async def arun(
        self,
        tool_input: str | dict[str, Any],
        verbose: bool | None = None,
        *,
        tool_call_id: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Run the tool asynchronously."""
        content: Any = None
        artifact: Any = None
        status: Literal["success", "error"] = "success"
        try:
            tool_args, tool_kwargs = self._to_args_and_kwargs(tool_input, tool_call_id)
            response = await self._arun(*tool_args, **tool_kwargs)
        except ValidationError as exc:
            if not self.handle_validation_error:
                raise
            content = self._handle_validation_error(exc)
            status = "error"
        except ToolException as exc:
            if not self.handle_tool_error:
                raise
            content = self._handle_error(exc)
            status = "error"
        else:
            if self.response_format == "content_and_artifact":
                if not isinstance(response, tuple) or len(response) != 2:
                    msg = (
                        "Since response_format='content_and_artifact' "
                        "a two-tuple of the message content and raw tool output is "
                        f"expected. Instead, generated response is of type: "
                        f"{type(response)}."
                    )
                    raise ValueError(msg)
                content, artifact = response
            else:
                content = response
        return _format_output(content, artifact, tool_call_id, self.name, status)

    def invoke(
        self,
        input: str | dict[str, Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the tool synchronously."""
        tool_call_id = None
        if isinstance(input, dict) and "args" in input and "name" in input:
            # A model ToolCall carries its id under "id"; accept the
            # "tool_call_id" spelling too so callers may pass either form.
            tool_call_id = input.get("id") or input.get("tool_call_id")
            input = input["args"]
        return self.run(input, tool_call_id=tool_call_id, **kwargs)

    async def ainvoke(
        self,
        input: str | dict[str, Any],
        config: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        """Invoke the tool asynchronously."""
        tool_call_id = None
        if isinstance(input, dict) and "args" in input and "name" in input:
            tool_call_id = input.get("id") or input.get("tool_call_id")
            input = input["args"]
        return await self.arun(input, tool_call_id=tool_call_id, **kwargs)

    def _run(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return self._run(*args, **kwargs)


def _format_output(
    content: Any,
    artifact: Any,
    tool_call_id: str | None,
    name: str,
    status: str,
) -> ToolOutputMixin | Any:
    """Wrap raw tool output in a ``ToolMessage`` for a model tool call."""
    if (
        isinstance(content, list)
        and content
        and all(is_tool_output(item) for item in content)
    ):
        return content
    if is_tool_output(content) or tool_call_id is None:
        return content
    normalized_content = _normalize_message_content(content)
    content = _stringify(content) if normalized_content is None else normalized_content
    return ToolMessage(
        content,
        artifact=artifact,
        tool_call_id=tool_call_id,
        name=name,
        status=status,
    )


def _normalize_message_content(obj: Any) -> str | list[Any] | None:
    """Coerce valid message content, or return ``None`` to signal stringification."""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, Sequence) and all(_is_message_content_block(e) for e in obj):
        return list(obj)
    return None


def _is_message_content_block(obj: Any) -> bool:
    """Check whether an object is a supported message content block."""
    if isinstance(obj, str):
        return True
    if isinstance(obj, dict):
        return obj.get("type", None) in TOOL_MESSAGE_BLOCK_TYPES
    return False


def _stringify(content: Any) -> str:
    """Convert content to a string, preferring JSON."""
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:  # noqa: BLE001 - match upstream's fallback contract
        return str(content)


class StructuredTool(BaseTool):
    """Tool that can operate on any number of inputs."""

    func: Callable[..., Any] | None = None
    coroutine: Callable[..., Awaitable[Any]] | None = None

    @classmethod
    def from_function(
        cls,
        func: Callable[..., Any] | None = None,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
        name: str | None = None,
        description: str | None = None,
        return_direct: bool = False,
        args_schema: ArgsSchema | None = None,
        infer_schema: bool = True,
        *,
        response_format: Literal["content", "content_and_artifact"] = "content",
        parse_docstring: bool = False,
        error_on_invalid_docstring: bool = False,
        **kwargs: Any,
    ) -> StructuredTool:
        """Create a tool from a function or coroutine."""
        source_function = func if func is not None else coroutine
        if source_function is None:
            msg = "Function and/or coroutine must be provided"
            raise ValueError(msg)
        name = name or source_function.__name__
        if args_schema is None and infer_schema:
            args_schema = create_schema_from_function(
                name,
                source_function,
                parse_docstring=parse_docstring,
                error_on_invalid_docstring=error_on_invalid_docstring,
                filter_args=_filter_schema_args(source_function),
            )
        description_ = description
        if description is None and not parse_docstring:
            description_ = source_function.__doc__ or None
        if description_ is None and args_schema:
            if isinstance(args_schema, type) and issubclass(args_schema, BaseModel):
                description_ = args_schema.__doc__
                if description_ and "A base class for creating Pydantic models" in description_:
                    description_ = ""
                elif not description_:
                    description_ = None
            elif isinstance(args_schema, dict):
                description_ = args_schema.get("description")
            else:
                msg = (
                    f"Invalid args_schema: expected BaseModel or dict, got {args_schema}"
                )
                raise TypeError(msg)
        if description_ is None:
            msg = "Function must have a docstring if description not provided."
            raise ValueError(msg)
        if description is None:
            description_ = textwrap.dedent(description_).strip()
        description_ = f"{description_.strip()}"
        return cls(
            name=name,
            func=func,
            coroutine=coroutine,
            args_schema=args_schema,
            description=description_,
            return_direct=return_direct,
            response_format=response_format,
            **kwargs,
        )

    @functools.cached_property
    def _injected_args_keys(self) -> frozenset[str]:
        fn = self.func or self.coroutine
        if fn is None:
            return _EMPTY_SET
        return frozenset(
            key
            for key, type_ in _get_type_hints_or_empty(fn).items()
            if _is_injected_arg_type(type_)
        )

    def _run(
        self,
        *args: Any,
        config: Any | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.func:
            return self.func(*args, **kwargs)
        msg = "StructuredTool does not support sync invocation."
        raise NotImplementedError(msg)

    async def _arun(
        self,
        *args: Any,
        config: Any | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> Any:
        if self.coroutine:
            return await self.coroutine(*args, **kwargs)
        return await super()._arun(*args, config=config, run_manager=run_manager, **kwargs)


def _filter_schema_args(func: Callable[..., Any]) -> list[str]:
    filter_args = list(FILTERED_ARGS)
    if config_param := _get_runnable_config_param(func):
        filter_args.append(config_param)
    return filter_args


def tool(
    name_or_callable: str | Callable[..., Any] | None = None,
    runnable: Callable[..., Any] | None = None,
    *args: Any,
    description: str | None = None,
    return_direct: bool = False,
    args_schema: ArgsSchema | None = None,
    infer_schema: bool = True,
    response_format: Literal["content", "content_and_artifact"] = "content",
    parse_docstring: bool = False,
    error_on_invalid_docstring: bool = True,
    extras: dict[str, Any] | None = None,
) -> BaseTool | Callable[[Callable[..., Any]], BaseTool]:
    """Convert a Python function into a tool, as a decorator or a direct call."""
    if args:
        msg = "Positional arguments are not supported when using the tool decorator."
        raise TypeError(msg)
    if callable(name_or_callable) and not isinstance(name_or_callable, str):
        return _create_tool(
            name_or_callable,
            None,
            name=None,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
            extras=extras,
        )
    if runnable is not None:
        if not isinstance(name_or_callable, str):
            msg = "Tool name must be a string when a runnable is provided."
            raise TypeError(msg)
        return _create_tool(
            runnable,
            None,
            name=name_or_callable,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
            extras=extras,
        )

    def decorator(func: Callable[..., Any]) -> BaseTool:
        return _create_tool(
            func,
            None,
            name=name_or_callable,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
            extras=extras,
        )

    return decorator


def _create_tool(
    func: Callable[..., Any],
    coroutine: Callable[..., Awaitable[Any]] | None,
    *,
    name: str | None,
    description: str | None,
    return_direct: bool,
    args_schema: ArgsSchema | None,
    infer_schema: bool,
    response_format: Literal["content", "content_and_artifact"],
    parse_docstring: bool,
    error_on_invalid_docstring: bool,
    extras: dict[str, Any] | None,
) -> BaseTool:
    is_async = inspect.iscoroutinefunction(func)
    kwargs: dict[str, Any] = {}
    if extras is not None:
        kwargs["metadata"] = extras
    if is_async:
        return StructuredTool.from_function(
            coroutine=func,
            name=name,
            description=description,
            return_direct=return_direct,
            args_schema=args_schema,
            infer_schema=infer_schema,
            response_format=response_format,
            parse_docstring=parse_docstring,
            error_on_invalid_docstring=error_on_invalid_docstring,
            **kwargs,
        )
    return StructuredTool.from_function(
        func=func,
        name=name,
        description=description,
        return_direct=return_direct,
        args_schema=args_schema,
        infer_schema=infer_schema,
        response_format=response_format,
        parse_docstring=parse_docstring,
        error_on_invalid_docstring=error_on_invalid_docstring,
        **kwargs,
    )
