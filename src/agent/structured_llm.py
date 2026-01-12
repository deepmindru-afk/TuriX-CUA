from __future__ import annotations
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.controller.views import *

# ---------------------------------------------------------------------------
# DISCRIMINATED UNION FOR A SINGLE ACTION ITEM
# ---------------------------------------------------------------------------

class ActionItem(BaseModel):
    """Exactly one of the fields must be populated to specify the concrete action."""
    model_config = ConfigDict(exclude_none=True)
    done: Optional[NoParamsAction] = None
    input_text: Optional[InputTextAction] = None
    open_app: Optional[OpenAppAction] = None
    run_apple_script: Optional[AppleScriptAction] = None
    Hotkey: Optional[PressAction] = None
    multi_Hotkey: Optional[PressCombinedAction] = None
    RightSingle: Optional[RightClickPixel] = None
    Click: Optional[LeftClickPixel] = None
    Drag: Optional[DragAction] = None
    move_mouse: Optional[MoveToAction] = None
    scroll_up: Optional[ScrollUpAction] = None
    scroll_down: Optional[ScrollDownAction] = None
    record_info: Optional[RecordAction] = None
    wait: Optional[NoParamsAction] = None

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @field_validator("wait", mode="before")
    def fix_empty_string(cls, v):
        if v == "" or v is None:
            return {}
        if not isinstance(v, dict):
            return {}
        return v

# ---------------------------------------------------------------------------
# CURRENT-STATE SUB-MODEL
# ---------------------------------------------------------------------------

class Analysis(BaseModel):
    analysis: str = Field(..., description="Detailed analysis of how the current state matches the expected state.")

class CurrentState(BaseModel):
    step_evaluate: str = Field(..., description="Success/Failed (based on step completion)")
    ask_human: str = Field(
        ...,
        description=(
            "Describe what you want user to do or No (No if nothing to ask for comfirmation. "
            "If something is unclear, ask the user for confirmation, like ask the user to login, or comfirm preference.)"
        ),
    )
    next_goal: str = Field(
        ..., description="Goal of this step based on actions, ONLY DESCRIBE THE EXPECTED ACTIONS RESULT OF THIS STEP"
    )

# ---------------------------------------------------------------------------
# AGENT OUTPUT MODELS
# ---------------------------------------------------------------------------

class MemoryOutput(BaseModel):
    summary: str = Field(..., description="Brief summary to remember for future steps.")

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @property
    def content(self) -> str:
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

    @property
    def parsed(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_unset=True)

class BrainOutput(BaseModel):
    analysis: Analysis
    current_state: CurrentState

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @property
    def content(self) -> str:
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

    @property
    def parsed(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_unset=True)

class ActorOutput(BaseModel):
    action: List[ActionItem] = Field(
        ...,
        min_items=0,
        max_items=10,
        description="Ordered list of 0-10 actions for this step.",
    )

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @property
    def content(self) -> str:
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

    @property
    def parsed(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True, exclude_unset=True)

class Step(BaseModel):
    step_id: str = Field(..., pattern=r"^Step \d+$")
    description: Optional[str]

class PlannerOutput(BaseModel):
    step_by_step_plan: List[Step] = Field(
        ...,
        min_items=1,
        description="Ordered high-level plan objects, each must start with 'Step N'.",
    )

    @property
    def content(self) -> str:
        lines = []
        for step in self.step_by_step_plan:
            lines.append(f"{step.step_id}: {step.description}")
        return "\n".join(lines)

__all__ = [
    "BrainOutput",
    "ActorOutput",
    "PlannerOutput",
    "MemoryOutput",
]
