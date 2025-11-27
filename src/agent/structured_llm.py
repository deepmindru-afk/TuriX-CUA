from __future__ import annotations
import re
import json
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, ConfigDict, field_validator
from src.controller.views import *
from pydantic.v1 import validator
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
    record_info: Optional[NoParamsAction] = None
    wait: Optional[NoParamsAction] = None

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"
    
    @field_validator("wait", "record_info", mode="before")
    def fix_empty_string(cls, v):
        if v == "" or v is None:
            return {}             # an empty dict is valid input for NoParamsAction
        if not isinstance(v, dict):
            return {}
        return v

# ---------------------------------------------------------------------------
# CURRENT‑STATE SUB‑MODEL
# ---------------------------------------------------------------------------

class Analysis(BaseModel):
    analyzation: str = Field(..., description="Detailed analysis of how the current state matches the expected state.")

class CurrentState(BaseModel):
    evaluation_previous_goal: str = Field(..., description="Success/Failed (based on step completion)")
    ask_human: str = Field(..., description="Describe what you want user to do or No (No if nothing to ask for comfirmation. If something is unclear, ask the user for confirmation, like ask the user to login, or comfirm preference.)")
    next_goal: str = Field(..., description="Goal of this step based on actions, ONLY DESCRIBE THE EXPECTED ACTIONS RESULT OF THIS STEP")


# ---------------------------------------------------------------------------
# AGENT STEP OUTPUT (MAIN MODEL)
# ---------------------------------------------------------------------------

class BrainOutput(BaseModel):
    """Schema for the agent's per‑step output.

    - ``action``: list of actions the agent should perform in order. Multiple actions
      are allowed in a single step.
    - ``current_state``: diagnostic information that supervisors/evaluators can use.
    """
    analysis: Analysis
    current_state: CurrentState

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @property
    def content(self) -> str:
        """
        Returns a JSON-formatted string representation of the instance,
        allowing access via the `.content` attribute.
        """
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

    @property
    def parsed(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the instance,
        facilitating direct access to structured data.
        """
        return self.model_dump(exclude_none=True, exclude_unset=True)

class ActorOutput(BaseModel):
    action: List[ActionItem] = Field(
        ...,
        min_items=0,
        max_items=10,                     # ← hard limit
        description="Ordered list of 0-10 actions for this step."
    )

    def __repr__(self) -> str:
        non_none = self.model_dump(exclude_none=True)
        field_strs = ", ".join(f"{k}={v!r}" for k, v in non_none.items())
        return f"{self.__class__.__name__}({field_strs})"

    @property
    def content(self) -> str:
        """
        Returns a JSON-formatted string representation of the instance,
        allowing access via the `.content` attribute.
        """
        return self.model_dump_json(exclude_none=True, exclude_unset=True)

    @property
    def parsed(self) -> Dict[str, Any]:
        """
        Returns the dictionary representation of the instance,
        facilitating direct access to structured data.
        """
        return self.model_dump(exclude_none=True, exclude_unset=True)


__all__ = [
    "BrainOutput",
    "ActorOutput",
]
