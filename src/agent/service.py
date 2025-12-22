from __future__ import annotations
import asyncio
import base64
import io
import json
import logging
import os
import uuid
from pathlib import Path
import Quartz
from typing import Any, Callable, Dict, List, Optional, Type, TypeVar
import re
from datetime import datetime
from dotenv import load_dotenv
from langchain_core.language_models.chat_models import BaseChatModel
from typing import Type
from collections import OrderedDict
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI, AzureChatOpenAI          # OpenAI endpoints
from langchain_anthropic import ChatAnthropic                     # Claude
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.messages import (
    BaseMessage,
)

from lmnr import observe
from openai import RateLimitError
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ValidationError
from src.agent.message_manager.service import MessageManager
from src.agent.prompts import (
    BrainPrompt_turix,
    ActorPrompt_turix,
    MemoryPrompt,
    PlannerPrompt,
)
from src.agent.views import (
    ActionResult,
    AgentError,
    AgentHistory,
    AgentHistoryList,
    AgentOutput,
    AgentStepInfo,
    AgentBrain
)
from src.agent.planner_service import Planner
from src.controller.service import Controller
from src.mac.tree import MacUITreeBuilder
from src.utils import time_execution_async
from src.agent.output_schemas import OutputSchemas
from src.agent.structured_llm import *

load_dotenv()
logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)

def screenshot_to_dataurl(screenshot):
    img_byte_arr = io.BytesIO()
    screenshot.save(img_byte_arr, format='PNG')
    base64_encoded = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')
    return f'data:image/png;base64,{base64_encoded}'

def to_structured(llm: BaseChatModel, Schema, Structured_Output) -> BaseChatModel:
    """
    Wrap *any* LangChain chat model with the right structured-output mechanism:

    • ChatOpenAI / AzureChatOpenAI  → bind(response_format=…)      (OpenAI style)
    • ChatAnthropic / ChatGoogleGenerativeAI → with_structured_output(…) (Claude/Gemini style)
    • anything else → returned unchanged
    """
    OPENAI_CLASSES: tuple[Type[BaseChatModel], ...] = (ChatOpenAI, AzureChatOpenAI)
    ANTHROPIC_OR_GEMINI: tuple[Type[BaseChatModel], ...] = (
        ChatAnthropic,
        ChatGoogleGenerativeAI,
    )

    if isinstance(llm, OPENAI_CLASSES):
        # OpenAI only: use the response_format param with your flattened schema
        return llm.bind(response_format=Schema)

    if isinstance(llm, ANTHROPIC_OR_GEMINI):
        # Claude & Gemini accept any schema textually → keep the nice Pydantic model
        return llm.with_structured_output(Structured_Output)

    # Fallback: no structured output
    return llm

class Agent:
    def __init__(
        self,
        task: str,
        brain_llm: BaseChatModel,
        actor_llm: BaseChatModel,
        memory_llm: BaseChatModel,
        short_memory_len : int,
        controller: Controller = Controller(),
        use_ui = False,
        use_search: bool = True,
        planner_llm: Optional[BaseChatModel] = None,
        save_brain_conversation_path: Optional[str] = None,
        save_brain_conversation_path_encoding: Optional[str] = 'utf-8',
        save_actor_conversation_path: Optional[str] = None,
        save_actor_conversation_path_encoding: Optional[str] = 'utf-8',
        max_failures: int = 5,
        memory_budget: int = 500,
        retry_delay: int = 10,
        max_input_tokens: int = 32000,
        resume = False,
        include_attributes: list[str] = [
            'title',
            'type',
            'name',
            'role',
            'tabindex',
            'aria-label',
            'placeholder',
            'value',
            'alt',
            'aria-expanded',
        ],
        max_error_length: int = 400,
        max_actions_per_step: int = 10,

        register_new_step_callback: Callable[['str', 'AgentOutput', int], None] | None = None,
        register_done_callback: Callable[['AgentHistoryList'], None] | None = None,
        tool_calling_method: Optional[str] = 'auto',
        agent_id: Optional[str] = None,
    ):
        self.wait_this_step = False
        if agent_id:
            self.agent_id = agent_id
        else:
            self.agent_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.task = task
        self.memory_budget = memory_budget  # Max number of previous actions to keep in memory
        self.original_task = task
        self.resume = resume
        self.memory_llm = to_structured(memory_llm, OutputSchemas.MEMORY_RESPONSE_FORMAT, MemoryOutput)
        self.brain_llm = to_structured(brain_llm, OutputSchemas.BRAIN_RESPONSE_FORMAT, BrainOutput)
        self.actor_llm = to_structured(actor_llm, OutputSchemas.ACTION_RESPONSE_FORMAT, ActorOutput)
        self.planner_llm_raw = planner_llm
        self.planner_llm = to_structured(planner_llm, OutputSchemas.PLANNER_RESPONSE_FORMAT, PlannerOutput)

        self.save_actor_conversation_path = save_actor_conversation_path
        self.save_actor_conversation_path_encoding = save_actor_conversation_path_encoding

        self.save_brain_conversation_path = save_brain_conversation_path
        self.save_brain_conversation_path_encoding = save_brain_conversation_path_encoding

        self.include_attributes = include_attributes
        self.max_error_length = max_error_length
        self.screenshot_annotated = None
        self.short_memory_len = short_memory_len
        self.max_input_tokens = max_input_tokens
        self.save_temp_file_path = os.path.join(os.path.dirname(__file__), 'temp_files')
        self.use_ui = use_ui
        self.use_search = use_search
        self.next_goal = ''
        self.brain_thought = ''

        self.mac_tree_builder = MacUITreeBuilder()
        self.controller = controller
        self.max_actions_per_step = max_actions_per_step
        self.last_step_action = None
        self.goal_action_memory = OrderedDict()

        self.last_goal = None
        self.brain_context = OrderedDict()
        self.status = "success"
        # Setup dynamic Action Model
        self._setup_action_models()
        # self._set_model_names()
        if self.planner_llm:
            self.planner = Planner(
                planner_llm=self.planner_llm,
                task=self.task,
                max_input_tokens=self.max_input_tokens,
                search_llm=self.planner_llm_raw,
                use_search=self.use_search,
            )

        # self.tool_calling_method = self.set_tool_calling_method(tool_calling_method)
        self.initiate_messages()
        self._last_result = None

        self.register_new_step_callback = register_new_step_callback
        self.register_done_callback = register_done_callback

        # Agent run variables
        self.history: AgentHistoryList = AgentHistoryList(history=[])
        self.n_steps = 1
        self.consecutive_failures = 0
        self.max_failures = max_failures
        self.retry_delay = retry_delay
        self._paused = False
        self._stopped = False
        self.brain_memory = ''
        self.infor_memory = []
        self.last_pid = None
        self.ask_for_help = False

        if self.resume and not agent_id:
            raise ValueError("Agent ID is required for resuming a task.")
        self.save_temp_file_path = os.path.join(self.save_temp_file_path, f"{self.agent_id}")
        

    def _set_model_names(self) -> None:
        self.chat_model_library = self.llm.__class__.__name__
        if hasattr(self.llm, 'model_name'):
            self.model_name = self.llm.model_name  # type: ignore
        elif hasattr(self.llm, 'model'):
            self.model_name = self.llm.model  # type: ignore
        else:
            self.model_name = 'Unknown'

    def set_tool_calling_method(self, tool_calling_method: Optional[str]) -> Optional[str]:
        if tool_calling_method == 'auto':
            if self.chat_model_library == 'ChatGoogleGenerativeAI':
                return None
            elif self.chat_model_library == 'ChatOpenAI':
                return 'function_calling'
            elif self.chat_model_library == 'AzureChatOpenAI':
                return 'function_calling'
            else:
                return None

    def _setup_action_models(self) -> None:
        """Setup dynamic action models from controller's registry"""
        self.ActionModel = self.controller.registry.create_action_model()
        self.AgentOutput = AgentOutput.type_with_custom_actions(self.ActionModel)

    def get_last_pid(self) -> Optional[int]:
        latest_pid = self.last_pid
        if self._last_result:
            for r in self._last_result:
                if r.current_app_pid:
                    latest_pid = r.current_app_pid
        return latest_pid

    async def _summarise_memory(self) -> None:
        """
        Summarise the current memory to reduce its size
        """
        memory_content = [
                    {
                        "type": "text",
                        "content": (
                            self.brain_memory
                        )
                    }
                ]
        self.memory_message_manager._remove_last_state_message()
        self.memory_message_manager._remove_last_AIntool_message()
        self.memory_message_manager.add_state_message(memory_content)
        memory_messages = self.memory_message_manager.get_messages()
        response = await self.memory_llm.ainvoke(memory_messages)
        memory_text = str(response.content)
        cleaned_memory_response = re.sub(r"^```(json)?", "", memory_text.strip())
        cleaned_memory_response = re.sub(r"```$", "", cleaned_memory_response).strip()
        logger.debug(f"[Memory] Raw text: {cleaned_memory_response}")
        parsed = json.loads(cleaned_memory_response)
        memory = parsed["summary"]
        self.brain_memory = 'The concise memory summary is:\n'
        self.brain_memory += memory
        self.brain_memory += 'The detail steps info are:\n'

    async def _update_memory(self) -> None:
        """
        Update memory content
        """

        memory_content = []
        sorted_steps = sorted(self.brain_context.keys(), reverse=True)
        logger.debug(f"all memory: {self.brain_context}")
        analysis = self.brain_context[sorted_steps[0]]['analysis'] if sorted_steps else None
        current_state = self.brain_context[sorted_steps[0]]['current_state'] if sorted_steps else None
        logger.debug(f"current_state: {current_state}")
        step_goal = current_state['next_goal'] if current_state else None
        logger.debug(f"step_goal: {step_goal}")
        evaluation = current_state['step_evaluate'] if current_state else None
        ask_human = current_state['ask_human'] if current_state else None

        if len(self.brain_memory) > self.memory_budget:
            await self._summarise_memory()

        line = f"Step {sorted_steps[0]} | Eval: {evaluation} | Goal: {step_goal}"
        self.brain_memory = "\n".join([ln for ln in [self.brain_memory, line] if ln]).strip()

        
    def save_memory(self) -> None:
        """
        Save the current memory to a file.
        """
        if not self.save_temp_file_path:
            return
        data = {
            "pid": self.get_last_pid(),
            "task": self.task,
            "next_goal": self.next_goal,
            "last_step_action": self.last_step_action,
            "infor_memory": self.infor_memory,
            'brain_context': self.brain_context,
            "step": self.n_steps
        }
        file_name = os.path.join(self.save_temp_file_path, f"memory.jsonl")
        os.makedirs(os.path.dirname(file_name), exist_ok=True) if os.path.dirname(file_name) else None
        with open(file_name, "w", encoding=self.save_brain_conversation_path_encoding) as f:
            if os.path.getsize(file_name) > 0:
                f.truncate(0)
            f.write(json.dumps(data, ensure_ascii=False, default=lambda o: list(o) if isinstance(o, set) else o) + "\n")

    async def load_memory(self) -> None:
        """
        Load the current memory from a file.
        """
        if not self.save_temp_file_path:
            return
        file_name = os.path.join(self.save_temp_file_path, f".jsonl")
        if os.path.exists(file_name):
            with open(file_name, "r", encoding=self.save_brain_conversation_path_encoding) as f:
                lines = f.readlines()
            if len(lines) >= 1:
                data = json.loads(lines[-1])
                self.task = data.get("task", "")
                self.last_pid = data.get("pid", None)
                self.infor_memory = data.get("infor_memory", [])
                # self.state_memory = data.get("state_memory", None)
                self.brain_context = data.get("brain_context", OrderedDict())
                if self.brain_context:
                    self.brain_context = OrderedDict({int(k): v for k, v in self.brain_context.items()})
                await self._update_memory()
                self.last_step_action = data.get("last_step_action", None)
                self.next_goal = data.get("next_goal", "")
                self.n_steps = data.get("step", 1)
                logger.info(f"Loaded memory from {file_name}")

    @time_execution_async('--brain_step')
    async def brain_step(self,) -> dict:
        step_id = self.n_steps
        logger.info(f"\n📍 Step {self.n_steps}")
        prev_step_id = step_id - 1
        try:
            self.previous_screenshot = self.screenshot_annotated
            screenshot = self.mac_tree_builder.capture_screenshot()
            self.screenshot_annotated = screenshot
            screenshot.save(f'images/screenshot_{self.n_steps}.png')
            current_screenshot_path = f'images/screenshot_{self.n_steps}.png'
            if self.screenshot_annotated:
                screenshot_dataurl = screenshot_to_dataurl(self.screenshot_annotated)
            if self.previous_screenshot:
                previous_screenshot_dataurl = screenshot_to_dataurl(self.previous_screenshot)
            # Build content for state message
            if step_id >= 2:
                state_content = [
                    {
                        "type": "text",
                        "content": (
                            f"Previous step is {prev_step_id}.\n\n"
                            f"Necessary information remembered is:\n{self.infor_memory}\n\n"
                            f"Previous Actions Short History:\n{self.brain_memory}\n\n"
                        )
                    }
                ]
                if previous_screenshot_dataurl:
                    state_content.append({
                        "type": "image_url",
                        "image_url": {"url": previous_screenshot_dataurl},
                    })
                if screenshot_dataurl:
                    state_content.append({
                        "type": "image_url",
                        "image_url": {"url": screenshot_dataurl},
                    })
            else:
                state_content = [
                    {
                        "type": "text",
                        "content": (
                            f"This is the first step.\n\n"
                            f"You should provide a JSON with a well-defined goal based on images information. The other fields should be default value."
                        )
                    }
                ]
                if screenshot_dataurl:
                    state_content.append({
                        "type": "image_url",
                        "image_url": {"url": screenshot_dataurl},
                    })
            
            self.brain_message_manager._remove_last_state_message()
            self.brain_message_manager._remove_last_AIntool_message()
            self.brain_message_manager.add_state_message(state_content)
            brain_messages = self.brain_message_manager.get_messages()
            
            response = await self.brain_llm.ainvoke(brain_messages)
            brain_text = str(response.content)
            cleaned_brain_response = re.sub(r"^```(json)?", "", brain_text.strip())
            cleaned_brain_response = re.sub(r"```$", "", cleaned_brain_response).strip()
            logger.debug(f"[Brain] Raw text: {cleaned_brain_response}")
            parsed = json.loads(cleaned_brain_response)
            self._save_brain_conversation(brain_messages, parsed, step=self.n_steps)
            self.brain_context[self.n_steps] = parsed
            self.next_goal = parsed['current_state']['next_goal']
            self.brain_thought = parsed["analysis"]
            self.current_state = parsed['current_state']

        except Exception as e:
            logger.exception("[Brain] Unexpected error in brain_step.")
            return {"Brain_text": {"step_evaluate": "unknown", "reason": str(e)}}

    @time_execution_async("--actor_step")
    async def actor_step(self, step_info: Optional[AgentStepInfo] = None) -> None:
        step_id = self.n_steps
        state = "" # Default value
        model_output = None
        result: list[ActionResult] = []
        prev_step_id = step_id - 1
        try:
            #---------------------------
            # 1) Build the UI tree and capture a screenshot
            #---------------------------
            logger.debug(f'Last PID: {self.last_pid}')
            if self.use_ui:
                self.last_pid = self.get_last_pid()
                root = await self.mac_tree_builder.build_tree(self.last_pid)
                state = root._get_visible_clickable_elements_string() if root else "No UI tree found."
            else:
                state = ''
            self.save_memory()
            # ---------------------------
            # 3) Define the input message for the core agent
            # ---------------------------
            if self.n_steps >= 2:
                if self.use_ui:
                    state_content = [
                        {
                            "type": "text",
                            "content": f"Previous step is {prev_step_id}.\n\nYour goal to achieve in this step is: {self.next_goal}\n\n"
                                        f"Analysis to the current screen is: {self.brain_thought}.\n\nNecessary information remembered is: {self.infor_memory}\n\nCurrent UI state:\n{state}"
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_to_dataurl(self.screenshot_annotated)},
                        }
                    ]
                else:
                    state_content = [
                        {
                            "type": "text",
                            "content": (
                                f"Necessary information remembered is: {self.infor_memory}\n\n"
                                f"Analysis to the current screen is: {self.brain_thought}.\n\n"
                                f"Your goal to achieve in this step is: {self.next_goal}\n\n"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": screenshot_to_dataurl(self.screenshot_annotated)},
                        }
                    ]
            else:
                state_content = [
                    {
                        "type": "text",
                        "content": f"Analysis to the current screen is: {self.brain_thought}. Your goal to achieve in this step is: {self.next_goal}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": screenshot_to_dataurl(self.screenshot_annotated)},
                    }
                ]
            self.actor_message_manager._remove_last_AIntool_message()
            self.actor_message_manager._remove_last_state_message()
            self.actor_message_manager.add_state_message(state_content, step_info = step_info)
            
            actor_messages = self.actor_message_manager.get_messages()
            model_output, raw = await self.get_next_action(actor_messages)

            self.last_goal = self.next_goal
            if self.register_new_step_callback:
                self.register_new_step_callback(state, model_output, self.n_steps)
            self._save_actor_conversation(actor_messages, model_output, step=self.n_steps)

            self.actor_message_manager._remove_last_state_message()
            self.actor_message_manager.add_model_output(model_output)
            
            self.last_step_action = [action.model_dump(exclude_unset=True) for action in model_output.action] if model_output else []
            # join the self.state_memory and the self.last_goal

            result = await self.controller.multi_act(
                model_output.action,
                self.mac_tree_builder,
                action_valid=True
            )
            self._last_result = result
            if self.use_ui:
                for i in range(len(model_output.action)):
                    if 'open_app' in str(model_output.action[i]):
                        logger.debug(f'Found open_app action, building the tree again')
                        await self.mac_tree_builder.build_tree(self.get_last_pid())
            if len(self.last_step_action) == 0:
                self.wait_this_step = True
            elif 'wait' in str(self.last_step_action[0]):
                self.wait_this_step = True
            else:
                self.wait_this_step = False
            if self.last_step_action and not self.wait_this_step:

                await self._update_memory()
                self.save_memory()

        except Exception as e:
            result = await self._handle_step_error(e)
            self._last_result = result
        finally:
            if result:
                self._make_history_item(model_output, state, result)
            if not self.wait_this_step:
                self.n_steps += 1

    async def _handle_step_error(self, error: Exception) -> list[ActionResult]:
        include_trace = logger.isEnabledFor(logging.DEBUG)
        error_msg = AgentError.format_error(error, include_trace=include_trace)
        prefix = f'❌ Result failed {self.consecutive_failures + 1}/{self.max_failures} times:\n '

        if isinstance(error, (ValidationError, ValueError)):
            logger.error(f'{prefix}{error_msg}')
            if 'Max token limit reached' in error_msg:
                # Possibly reduce tokens from history
                self.actor_message_manager.max_input_tokens -= 500
                logger.info(f'Reducing agent max input tokens: {self.actor_message_manager.max_input_tokens}')
                self.actor_message_manager.cut_messages()
            elif 'Could not parse response' in error_msg:
                error_msg += '\n\nReturn a valid JSON object with the required fields.'
            self.consecutive_failures += 1

        elif isinstance(error, RateLimitError):
            logger.warning(f'{prefix}{error_msg}')
            await asyncio.sleep(self.retry_delay)
            self.consecutive_failures += 1

        else:
            logger.error(f'{prefix}{error_msg}')
            self.consecutive_failures += 1

        return [ActionResult(error=error_msg, include_in_memory=True)]

    def _make_history_item(
        self,
        model_output: AgentOutput | None,
        state: str,
        result: list[ActionResult],
    ) -> None:
        history_item = AgentHistory(
            model_output=model_output,
            result=result,
            state=state,
        )
        self.history.history.append(history_item)

    @time_execution_async('--get_next_action')
    async def get_next_action(self, input_messages: list[BaseMessage]) -> AgentOutput:
        """
        Build a 'structured_llm' approach on top of self.llm. 
        Using the dynamic self.AgentOutput
        """        
        response: dict[str, Any] = await self.actor_llm.ainvoke(input_messages)
        logger.debug(f'LLM response: {response}')
        record = str(response.content)

        output_dict = json.loads(record)
        for i in range(len(output_dict['action'])):
            outer_key = list(output_dict['action'][i].keys())[0]
            inner_value = output_dict['action'][i][outer_key]
            if outer_key == "record_info":
                information_stored = inner_value.get("text", "None")
                self.infor_memory.append({f'Step {self.n_steps}, the information stored is: {information_stored}'})
        parsed: AgentOutput | None = AgentOutput(action=output_dict['action'])

        self._log_response(parsed)
        return parsed, record
    

    def _log_response(self, response: AgentOutput) -> None:
        if 'Success' in self.current_state["step_evaluate"]:
            emoji = '✅'
        elif 'Failed' in self.current_state["step_evaluate"]:
            emoji = '❌'
        else:
            emoji = '🤷'
        logger.info(f'{emoji} Eval: {self.current_state["step_evaluate"]}')
        logger.info(f'🧠 Memory: {self.brain_context}')
        logger.info(f'🎯 Goal to achieve this step: {self.next_goal}')
        for i, action in enumerate(response.action):
            logger.info(f'🛠️  Action {i + 1}/{len(response.action)}: {action.model_dump_json(exclude_unset=True)}')
    
    def _save_brain_conversation(
        self,
        input_messages: list[BaseMessage],
        response: Any,
        step: int
    ) -> None:
        """
        Write all the Brain agent conversation (input messages + final AgentOutput)
        into a file: e.g. "brain_conversation_{step}.txt"
        """
        # If you do NOT want to save or no path provided, skip
        if not self.save_brain_conversation_path:
            return
        file_name = f"{self.save_brain_conversation_path}_brain_{step}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True) if os.path.dirname(file_name) else None

        with open(file_name, "w", encoding=self.save_brain_conversation_path_encoding) as f:
            # 1) Write input messages
            self._write_messages_to_file(f, input_messages)
            # 2) Write the final agent "response" (AgentOutput)
            if response is not None:
                self._write_response_to_file(f, response)

        logger.info(f"Brain conversation saved to: {file_name}")

    def _save_actor_conversation(
        self,
        input_messages: list[BaseMessage],
        response: Any,
        step: int
    ) -> None:
        """
        Write all the Actor agent conversation (input messages + final AgentOutput)
        into a file: e.g. "actor_conversation_{step}.txt"
        """
        # If you do NOT want to save or no path provided, skip
        if not self.save_actor_conversation_path:
            return
        file_name = f"{self.save_actor_conversation_path}_actor_{step}.txt"
        os.makedirs(os.path.dirname(file_name), exist_ok=True) if os.path.dirname(file_name) else None

        with open(file_name, "w", encoding=self.save_actor_conversation_path_encoding) as f:
            # 1) Write input messages
            self._write_messages_to_file(f, input_messages)
            # 2) Write the final agent "response" (AgentOutput)
            if response is not None:
                self._write_response_to_file(f, response)

        logger.info(f"Actor conversation saved to: {file_name}")

    def _write_messages_to_file(self, f: Any, messages: list[BaseMessage]) -> None:
        """
        For each message, write it out in a human-readable format.
        Or adapt your existing logic from _write_messages_to_file.
        """
        for message in messages:
            f.write(f"\n{message.__class__.__name__}\n{'-'*40}\n")
            if isinstance(message.content, list):
                for item in message.content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            txt = item.get('content') or item.get('text', '')
                            f.write(f"[Text Content]\n{txt.strip()}\n\n")
                        elif item.get('type') == 'image_url':
                            image_url = item['image_url']['url']
                            f.write(f"[Image URL]\n{image_url[:100]}...\n\n")
            else:
                # If it's a string or something else:
                f.write(f"{str(message.content)}\n\n")
            f.write('\n' + '='*60 + '\n')

    def _write_response_to_file(self, f: Any, response: Any) -> None:
        """
        If the AgentOutput is JSON-like, you can do:
        """
        f.write('RESPONSE\n')
        # If it's an AgentOutput, you might do:
        #   f.write(json.dumps(json.loads(response.model_dump_json(exclude_unset=True)), indent=2))
        # Otherwise just string-ify it:
        f.write(str(response) + '\n')

        f.write('\n' + '='*60 + '\n')

    def _log_agent_run(self) -> None:
        logger.info(f'🚀 Starting task: {self.task}')

    async def run(self, max_steps: int = 100) -> AgentHistoryList:
        try:
            self._log_agent_run()

            if self.planner_llm and not self.resume:
                await self.edit()

            for step in range(max_steps):
                if self.resume:
                    self.load_memory()
                    self.resume = False
                if self._too_many_failures():
                    break
                if not await self._handle_control_flags():
                    break

                await self.brain_step()
                await self.actor_step()

                if self.history.is_done():
                    logger.info('✅ Task completed successfully')
                    if self.register_done_callback:
                        self.register_done_callback(self.history)
                    break
                await asyncio.sleep(2)  # Wait before next step
            else:
                logger.info('❌ Failed to complete task in maximum steps')

            return self.history
        except Exception:
            logger.exception('Error running agent')
            raise

    async def edit(self):
        response = await self.planner.edit_task()
        self._set_new_task(response)

    PREFIX = "The overall user's task is: "
    SUFFIX = "The step by step plan is: "

    def _set_new_task(self, generated_plan: str) -> None:
        """
        Build the final task string:
            "The overall plan is: <original task>\n\n<generated plan>"
        and update every MessageManager in one go.
        """
        if generated_plan.startswith(self.PREFIX):
            final_task = generated_plan
        else:
            final_task = f"{self.PREFIX}{self.original_task}\n{self.SUFFIX}\n{generated_plan}"
        self.task = final_task
        self.initiate_messages()

    def _too_many_failures(self) -> bool:
        if self.consecutive_failures >= self.max_failures:
            logger.error(f'❌ Stopping due to {self.max_failures} consecutive failures')
            return True
        return False

    async def _handle_control_flags(self) -> bool:
        if self._stopped:
            logger.info('Agent stopped')
            return False

        while self._paused:
            await asyncio.sleep(0.2)
            if self._stopped:
                return False

        return True

    def save_history(self, file_path: Optional[str | Path] = None) -> None:
        if not file_path:
            file_path = 'AgentHistory.json'
        self.history.save_to_file(file_path)

    def initiate_messages(self):
        self.brain_message_manager = MessageManager(
            llm=self.brain_llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=BrainPrompt_turix, # Typically your SystemPrompt
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            give_task=True
        )
        self.actor_message_manager = MessageManager(
            llm=self.actor_llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=ActorPrompt_turix, # Typically your SystemPrompt
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            give_task=False
        )
        self.memory_message_manager = MessageManager(
            llm=self.memory_llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=MemoryPrompt, # Typically your SystemPrompt
            max_input_tokens=self.max_input_tokens,
            include_attributes=self.include_attributes,
            max_error_length=self.max_error_length,
            max_actions_per_step=self.max_actions_per_step,
            give_task=True
        )
