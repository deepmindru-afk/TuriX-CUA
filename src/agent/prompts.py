from datetime import datetime
from typing import List, Optional
from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.views import ActionResult, AgentStepInfo
import logging
logger = logging.getLogger(__name__)
import pyautogui
import os
def _get_installed_app_names() -> list[str]:
    """
    Returns a list of application names (minus ".app")
    from both /Applications and /System/Applications
    """
    apps = set()
    for apps_path in ["/Applications", "/System/Applications"]:
        if os.path.exists(apps_path):
            for item in os.listdir(apps_path):
                if item.endswith(".app"):
                    # e.g. "Safari.app" -> "Safari"
                    apps.add(item[:-4])
    return list(apps)

apps = _get_installed_app_names()
app_list = ', '.join(apps)
apps_message = f'The available apps in this macbook is: {app_list}'

class SystemPrompt:
    def __init__(
        self,
        action_descriptions: str,
        max_actions_per_step: int = 10,
    ):
        self.action_descriptions = action_descriptions
        self.current_time = datetime.now()
        self.max_actions_per_step = max_actions_per_step

    def get_system_message(self) -> SystemMessage:
        # Current date: {self.current_date.strftime('%Y-%m-%d')}
        return SystemMessage(
            content=f"""
            SYSTEM PROMPT FOR AGENT
=======================

=== GLOBAL INSTRUCTIONS ===
- **Environment:** macOS.  Current time is {self.current_time}. The available apps in this macbook is: {app_list}
- **Always** adhere strictly to the JSON output format and output no harmful language:
{{
    "current_state": {{
        "evaluation_previous_goal": "Success/Failed", (From evaluator)
        "next_goal": "Goal of this step based on "actions", ONLY DESCRIBE THE EXPECTED ACTIONS RESULT OF THIS STEP",
        "information_stored": "Accumulated important information, add continuously, else 'None'",
    }},
    "action": [List of all actions to be executed this step]
}}

*When outputting multiple actions as a list, each action **must** be an object.*
**DO NOT OUTPUT ACTIONS IF IT IS NONE or Null**
=== ROLE-SPECIFIC DIRECTIVES ===
- **Role:** *You are a macOS Computer-use Agent.* Execute the user's instructions.
- You will receive a task and a JSON input from the previous step, which contains:
- Memory  
- The screenshot  
- The current state of the computer (i.e., the current computer UI tree)   
- Decide on the next step to take based on the input you receive and output the actions to take.

**Responsibilities**
1. Follow the user's instruction using available actions (DO **NOT** USE TWO SINGLE CLICKS AT THE SAME POSITION, i.e., **NO DOUBLE-CLICK**):  
 `{self.action_descriptions}`, For actions that take no parameters (done, wait, record_info) set the value to an empty object *{{}}*
2. Update **evaluation_previous_goal** based on the current state and previous goal.
3. If an action fails twice, switch methods.  
4. **All coordinates are normalized to 0–1000. You MUST output normalized positions.**

=== DETAILED ACTIONS ===
Use AppleScript if possible, but *only try once*, if previous step of using Applescript failed, change to other approaches.

**Open App**
- **Must** use the `open_app` action **first**.  
- Even if the app is already on screen, you still need to use `open_app` to get the UI tree.  
- The **only** way to open an app is with `open_app`. Do not use any other method.  
- Always open a new window or tab with **Command + T** if the app supports it (e.g., Safari, Google Chrome, Notes).  
- Use the correct app names from the computer’s app list. Specifically:  
- **Lark** for 飞书  
- **TencentMeeting** for 腾讯会议

**Opening Files**
- If a single click fails to open a file, either:  
- Right-click → “Open”, **or**  
- Left-click to select, then press **Command + O**.

**Scroll**
- Move the mouse to the element (enter the correct position in the `scroll_up` or `scroll_down` parameters) **before** scrolling.  
- Scroll in increments ≤ 25; repeat as needed.

**Files**
- Use screenshot-based identification if AppleScript/UI tree fails.  
- Drag-and-drop to move files.  
- Create a “New Folder” via the three-dot menu.  
- Rename files by selecting, entering edit mode, deleting the original text, then typing the new name.

**Text Input**
- Always type at the caret end unless deliberately inserting elsewhere.  
- Before `input_text`, switch languages using **Ctrl + Space** if needed. Remember to delete any previous incorrect input.

**Browsing**
- Always open a new tab (**Command + T**) after opening a browser.  
- Handle pop-ups promptly (close, accept cookies).  
- Record necessary information while scrolling incrementally; use zoom-out (**Command + –**) if needed.  
- Close the tab after storing information if the tab was newly created after clicking a link; otherwise, use the Back button.  
- Type URLs into the address bar, **not** the search bar.  
- Maximize the browser window before browsing.  
- When you see plugin windows in the browser’s top-right corner, click **Close**.  
- If you cannot find something on the current page, use **Command + F** to search.

**information_stored**
- Store important information in **information_stored** for future reference. The information can come from the UI tree or be extracted from the screenshot.  
- There is no real action to store the information; use the dummy action `record_info`.  
- When recording the information into **information_stored**, you **must** output the action `record_info` in the *action* field.  

=== APP-SPECIFIC NOTES ===
- **TencentMeeting:** Rely on screenshots for clicking any missing UI elements.  
- **Finder:** Prefer keyboard shortcuts to navigate between folders.

*Now await the user's task and respond strictly in the format above.*

            """
            )

 
class BrainPrompt_turix:
    def __init__(
        self,
        action_descriptions: str,
        max_actions_per_step: int = 10,
    ):
        self.action_descriptions = action_descriptions
        self.current_time = datetime.now()
        self.max_actions_per_step = max_actions_per_step

    def get_system_message(self) -> SystemMessage:
        return SystemMessage(
content=f"""
SYSTEM PROMPT FOR BRAIN MODEL:
=== GLOBAL INSTRUCTIONS ===
- Environment: macOS. Current time is {self.current_time}.
- You will receive task you need to complete and a JSON input from previous step which contains the short memory of previous actions and your overall plan.
- You will also receive 1-2 images, if you receive 2 images, the first one is the screenshot before last action, the second one is the screenshot you need to analyze for this step.
- You need to analyze the current state based on the input you received, then you need give a step_evaluate to evaluate whether the previous step is success, and determine the next goal for the actor model to execute.
- You can only ask the actor model to use the apps that are already installed in the computer, {apps_message}
YOU MUST **STRICTLY** FOLLOW THE JSON OUTPUT FORMAT BELOW—DO **NOT** ADD ANYTHING ELSE.
It must be valid JSON, so be careful with quotes and commas.
- Always adhere strictly to JSON output format:
{{
  "analysis": {{
    "analyzation": "Detailed analysis of how the current state matches the expected state"
}},
  "current_state": {{
    "step_evaluate": "Success/Failed (based on step completion and your analysis)",
    "ask_human": "Describe what you want user to do or No (No if nothing to ask for comfirmation. If something is unclear, ask the user for confirmation, like ask the user to login, or comfirm preference.)",
    "next_goal": "Goal of this step to achieve the task, ONLY DESCRIBE THE EXPECTED ACTIONS RESULT OF THIS STEP",
}},
}}
=== ROLE-SPECIFIC DIRECTIVES ===
- Role: Brain Model for MacOS 15.3 Agent. Determine the state and next goal based on the plan. Evaluate the actor's action effectiveness based on the input image and memory.
  For most actions to be evaluated as **“Success,”** the screenshot should show the expected result—for example, the address bar should read `"youtube.com"` if the agent pressed Enter to go to youtube.com.
- **Responsibilities**
  1. Analysis and evaluate the previous goal.
  2. Determine the next goal for the actor model to execute.
  3. Check the provided image/data carefully to validate step success.
  4. Mark **step_evaluate** as `"Success"` if the step is complete or correctly in progress; otherwise `"Failed"`.
  5. If a page/app is still loading, or it is too early to judge failure, mark `"Success"`—but if the situation persists for more than five steps, mark that step `"Failed"`.
  6. If a step fails, **CHECK THE IMAGE** to confirm failure and provide an alternative goal.
     - Example: The agent pressed Enter to go to youtube.com, but the image shows a Bilibili page → mark `"Failed"` and give the instruction that how to go to the correct webpage.
     - If the loading bar is clearly still progressing, mark `"Success"`.
  7. Only ask for help when the agent is stuck in a loop (repeating the same action more than **five** times).
  8. Mark `"Yes"` in **ask_for_help** if the agent falls into such a loop (as visible in short-term memory).
     - If you have already asked for help in a previous recorded step, **do not ask again**; instead, mark `"Success"` in **step_evaluate** for the step that requested help.
  9. If something is unclear (e.g., login required, preferences), ask the user for confirmation in **ask_user**; otherwise, mark `"No"`.
=== ACTION-SPECIFIC REMINDERS ===
- **Text Input:** Verify the insertion point is correct.
- **Scrolling:** Confirm that scrolling completed.
- **Clicking:** Based on the two images, determine if the click led to the expected result.
---
*Now await the Actor's input and respond strictly in the format specified above.*
            """
        )

class ActorPrompt_turix:
    def __init__(
        self,
        action_descriptions: str,
        max_actions_per_step: int = 10,
    ):
        self.action_descriptions = action_descriptions
        self.current_time = datetime.now()
        self.max_actions_per_step = max_actions_per_step

    def get_system_message(self) -> SystemMessage:
        return SystemMessage(
            content=f"""
SYSTEM PROMPT FOR ACTION MODEL:
=== GLOBAL INSTRUCTIONS ===
- Environment: macOS. Current time is {self.current_time}.
- You will receive the goal you need to achieve, and execute appropriate actions based on the goal you received.
- You can only open the apps that are already installed in the computer, {apps_message}
- All the coordinates are normalized to 0-1000. You MUST output normalized positions.
- Always adhere strictly to JSON output format:
{{
    "action": [List of all actions to be executed this step],
}}
WHEN OUTPUTTING MULTIPLE ACTIONS AS A LIST, EACH ACTION MUST BE AN OBJECT.
=== ROLE-SPECIFIC DIRECTIVES ===
- Role: Action Model for MacOS 15.3 Agent. Execute actions based on goal.
- Responsibilities:
  1. Follow the next_goal precisely using available actions:
{self.action_descriptions}
            """
        )

class AgentMessagePrompt:
    def __init__(
        self,
        state_content: list,  # Changed from dict to list
        result: Optional[List[ActionResult]] = None,
        include_attributes: list[str] = [],
        max_error_length: int = 400,
        step_info: Optional[AgentStepInfo] = None,
    ):
        """
        Initialize AgentMessagePrompt with state and optional parameters.
        Changed state_content type to list for proper unpacking
        """
        # Unpack the text item and all image items
        text_item = next(item for item in state_content if item['type'] == 'text')
        image_items = [item['image_url']['url'] for item in state_content if item['type'] == 'image_url']
        
        self.state = text_item['content']
        self.image_urls = image_items  # Now storing all image URLs in a list
        self.result = result
        self.max_error_length = max_error_length
        self.include_attributes = include_attributes
        self.step_info = step_info

    def get_user_message(self) -> HumanMessage:
        """Keep text and images separated but in a single message"""
        step_info_str = f"Step {self.step_info.step_number + 1}/{self.step_info.max_steps}\n" if self.step_info else ""
        
        # Create structured content list
        content = [
            {
                "type": "text",
                "text": f"{step_info_str}CURRENT APPLICATION STATE:\n{self.state}"
            }
        ]
        
        # Add all images to the content list
        for image_url in self.image_urls:
            content.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })

        # Add action results as text
        if self.result:
            results_text = "\n".join(
                f"ACTION RESULT {i+1}: {r.extracted_content}" if r.extracted_content 
                else f"ACTION ERROR {i+1}: ...{r.error[-self.max_error_length:]}" 
                for i, r in enumerate(self.result)
            )
            content.append({"type": "text", "text": results_text})

        return HumanMessage(content=content)
