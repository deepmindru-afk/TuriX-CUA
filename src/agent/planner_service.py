from src.agent.message_manager.service import MessageManager
from src.agent.prompts import PlannerPrompt
from src.controller.service import Controller

import logging
from dotenv import load_dotenv
from langchain_core.messages import (
    HumanMessage,
)

load_dotenv()
logger = logging.getLogger(__name__)

class Planner:
    def __init__(self,
                 planner_llm,
                 task: str, 
                 max_input_tokens: int = 32000,
                 ):
        self.planner_llm = planner_llm
        self.controller = Controller()
        self.task = task
        self.max_input_tokens = max_input_tokens
        
        self.message_manager = MessageManager(
            llm=self.planner_llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=PlannerPrompt,
            max_input_tokens=self.max_input_tokens,
        )
        
    async def edit_task(self) -> str:
        if not self.planner_llm:
            return
        controller = Controller()
        planner_prompt = PlannerPrompt(controller.registry.get_prompt_description())
        system_message = planner_prompt.get_system_message().content
        content = f"""
                {system_message}
                Now, here is the task you need to break down:
                "{self.task}"
                Please follow the guidelines and provide the required JSON output.
                """
        response = await self.planner_llm.ainvoke([HumanMessage(content=content)])
        reply_text = response.content.strip()
        reply_norm = reply_text.upper()
        if "REFUSE TO MAKE PLAN" in reply_norm:
            logging.error("Planner refused. Aborting.")
            raise SystemExit(1) 
        return response.content