from __future__ import annotations

from typing import cast

from .._compat import get_model_fields
from .._models import BaseModel, build
from ..types.responses.response_input_item import (
    McpCall,
    ShellCall,
    McpListTools,
    ApplyPatchCall,
    LocalShellCall,
    ToolSearchCall,
    ShellCallOutput,
    ComputerCallOutput,
    FunctionCallOutput,
    McpApprovalRequest,
    ImageGenerationCall,
    McpApprovalResponse,
    ApplyPatchCallOutput,
    LocalShellCallOutput,
    ResponseOutputMessage,
    ResponseReasoningItem,
    ResponseCustomToolCall,
    ResponseComputerToolCall,
    ResponseFunctionToolCall,
    ResponseFunctionWebSearch,
    ResponseFileSearchToolCall,
    ResponseCompactionItemParam,
    ResponseCustomToolCallOutput,
    ResponseCodeInterpreterToolCall,
    ResponseToolSearchOutputItemParam,
)
from ..types.responses.response_input_item_param import ResponseInputItemParam

_INPUT_ITEM_MODEL_BY_TYPE: dict[str, type[BaseModel]] = {
    "apply_patch_call": ApplyPatchCall,
    "apply_patch_call_output": ApplyPatchCallOutput,
    "code_interpreter_call": ResponseCodeInterpreterToolCall,
    "compaction": ResponseCompactionItemParam,
    "computer_call": ResponseComputerToolCall,
    "computer_call_output": ComputerCallOutput,
    "custom_tool_call": ResponseCustomToolCall,
    "custom_tool_call_output": ResponseCustomToolCallOutput,
    "file_search_call": ResponseFileSearchToolCall,
    "function_call": ResponseFunctionToolCall,
    "function_call_output": FunctionCallOutput,
    "image_generation_call": ImageGenerationCall,
    "local_shell_call": LocalShellCall,
    "local_shell_call_output": LocalShellCallOutput,
    "mcp_approval_request": McpApprovalRequest,
    "mcp_approval_response": McpApprovalResponse,
    "mcp_call": McpCall,
    "mcp_list_tools": McpListTools,
    "message": ResponseOutputMessage,
    "reasoning": ResponseReasoningItem,
    "shell_call": ShellCall,
    "shell_call_output": ShellCallOutput,
    "tool_search_call": ToolSearchCall,
    "tool_search_output_item": ResponseToolSearchOutputItemParam,
    "web_search_call": ResponseFunctionWebSearch,
}


def output_as_input(output: list[BaseModel]) -> list[ResponseInputItemParam]:
    return [_output_item_as_input(item) for item in output]


def _output_item_as_input(item: BaseModel) -> ResponseInputItemParam:
    item_type = cast("str | None", getattr(item, "type", None))
    if item_type is None:
        raise TypeError(f"Response output item {item!r} does not define a type")

    target_model = _INPUT_ITEM_MODEL_BY_TYPE.get(item_type)
    if target_model is None:
        raise TypeError(f"Unsupported response output item type {item_type!r}")

    allowed_fields = set(get_model_fields(target_model))
    raw_item = item.to_dict(mode="json", exclude_unset=True, exclude_none=False)
    normalized_item = build(target_model, **{key: value for key, value in raw_item.items() if key in allowed_fields})
    return cast(ResponseInputItemParam, normalized_item.to_dict(mode="json", exclude_unset=True, exclude_none=False))
