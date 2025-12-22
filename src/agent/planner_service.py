import asyncio
import json
import logging
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.message_manager.service import MessageManager
from src.agent.prompts import PlannerPrompt
from src.controller.service import Controller

try:
    # Preferred package name (renamed from duckduckgo_search)
    from ddgs import DDGS  # type: ignore
    from ddgs.exceptions import DDGSException  # type: ignore
except ImportError:
    try:
        from duckduckgo_search import DDGS  # type: ignore
        from duckduckgo_search.exceptions import DuckDuckGoSearchException as DDGSException  # type: ignore
    except ImportError:
        DDGS = None
        DDGSException = Exception  # type: ignore

load_dotenv()
logger = logging.getLogger(__name__)
# Silence noisy logs from ddgs/primp to keep planner output clean.
logging.getLogger("primp").setLevel(logging.WARNING)
logging.getLogger("primp").propagate = False
logging.getLogger("ddgs").setLevel(logging.WARNING)


class Planner:
    def __init__(self,
                 planner_llm,
                 task: str,
                 max_input_tokens: int = 32000,
                 search_llm=None,
                 use_search: bool = True,
                 ):
        self.planner_llm = planner_llm
        self.controller = Controller()
        self.task = task
        self.max_input_tokens = max_input_tokens
        self.plan_list = []
        self._search_context: Optional[str] = None
        self.search_llm = search_llm
        self.use_search = use_search

        self.message_manager = MessageManager(
            llm=self.planner_llm,
            task=self.task,
            action_descriptions=self.controller.registry.get_prompt_description(),
            system_prompt_class=PlannerPrompt,
            max_input_tokens=self.max_input_tokens,
        )

    async def _decide_search_queries(self) -> List[str]:
        """
        Ask the planner model (raw, unstructured) to propose diverse search queries.
        Falls back to curated defaults when a helper LLM is unavailable.
        """
        if not self.use_search:
            return []

        fallback_queries = [
            "TuriX computer use agent capabilities",
            "AI agent automate computer tasks macOS",
            "Twitter X DM outreach best practices software sales",
            "Write persuasive AI software sales pitch example",
            "Social selling strategies for SaaS on Twitter",
            "TuriX case studies or customer success",
        ]
        if not self.search_llm:
            return fallback_queries

        try:
            prompt = SystemMessage(
                content=(
                    "Generate 3-6 concise web search queries for background research. "
                    "Cover: TuriX capabilities/value, AI computer-use agents, social selling for software (Twitter/X DMs), "
                    "examples of persuasive outreach copy, and user persona analysis tips. "
                    "Keep each query under 12 words, prefer English, avoid copying the whole user task text, "
                    "and ensure the set is diverse (no near-duplicates). "
                    "Respond strictly as a JSON array of strings."
                )
            )
            resp = await self.search_llm.ainvoke([prompt, HumanMessage(content=self.task)])
            text = getattr(resp, "content", "") or ""
            if isinstance(text, str):
                try:
                    data = json.loads(text)
                    if isinstance(data, list):
                        queries = [q.strip() for q in data if isinstance(q, str) and q.strip()]
                        # Deduplicate while preserving order
                        seen = set()
                        deduped = []
                        for q in queries:
                            if q not in seen:
                                deduped.append(q)
                                seen.add(q)
                        queries = deduped or fallback_queries
                        logger.info("Planner suggested search queries: %s", queries)
                        return queries
                except json.JSONDecodeError:
                    pass
            return fallback_queries
        except Exception as exc:
            logger.debug("Search query generation failed, using fallback queries: %s", exc, exc_info=True)
            return fallback_queries

    def _build_query_variants(self, query: str) -> List[tuple[str, Optional[str]]]:
        """
        Build a small set of query/backend combinations to increase the chance of results.
        """
        if not query:
            return []

        clean_query = query.strip()
        variants: List[tuple[str, Optional[str]]] = []

        # Primary attempt: full query with DuckDuckGo backend
        variants.append((clean_query, "duckduckgo"))

        # If the query is very long, try a truncated version (DuckDuckGo)
        if len(clean_query) > 256:
            variants.append((clean_query[:256], "duckduckgo"))

        # Fallbacks: let ddgs auto-select backend with original and truncated queries
        variants.append((clean_query, "auto"))
        if len(clean_query) > 256:
            variants.append((clean_query[:256], "auto"))

        return variants

    async def _fetch_search_results(self, query: str, max_results: int = 8) -> List[dict]:
        """
        Fetch DuckDuckGo search results in a background thread to avoid blocking the event loop.
        """
        if not query:
            return []
        if DDGS is None:
            logger.debug("duckduckgo_search not installed; skipping planner search context.")
            return []

        loop = asyncio.get_running_loop()

        def _search():
            try:
                with DDGS() as ddgs:
                    for q, backend in self._build_query_variants(query):
                        try:
                            results = list(ddgs.text(q, backend=backend, max_results=max_results))
                            if results:
                                logger.info("Planner search success (backend=%s, query=%r): %d results", backend, q, len(results))
                                return results
                            logger.info("Planner search empty (backend=%s, query=%r)", backend, q)
                        except DDGSException as exc:
                            logger.info("DuckDuckGo search (%s backend) returned no results: %s", backend, exc)
                        except Exception as exc:
                            logger.debug("DuckDuckGo search (%s backend) error: %s", backend, exc, exc_info=True)
                    return []
            except DDGSException as exc:
                logger.debug("DuckDuckGo search returned no results: %s", exc)
                return []
            except Exception as exc:
                logger.debug("DuckDuckGo search unexpected error: %s", exc, exc_info=True)
                return []

        try:
            return await loop.run_in_executor(None, _search)
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for query %s: %s", query, exc, exc_info=True)
            return []

    def _format_search_results(self, results: List[dict]) -> str:
        """
        Convert search results into a compact, readable text block.
        """
        lines = []
        for idx, item in enumerate(results, start=1):
            title = (item.get("title") or "No title").strip()
            snippet = (item.get("body") or "").strip().replace("\n", " ")
            href = (item.get("href") or "").strip()
            if len(snippet) > 200:
                snippet = snippet[:197] + "..."

            readable = f"{idx}. {title}"
            if snippet:
                readable += f" — {snippet}"
            if href:
                readable += f" (source: {href})"
            lines.append(readable)

        return "\n".join(lines)

    def _strip_source(self, line: str) -> str:
        """
        Remove source links/URLs from a summary line.
        """
        if "(source:" in line:
            line = line.split("(source:", 1)[0].rstrip()
        return line.rstrip(" -—·.")

    async def _get_search_context(self) -> str:
        """
        Run DuckDuckGo search once per planner instance and cache a readable summary.
        """
        if self._search_context is not None:
            return self._search_context
        if not self.use_search:
            self._search_context = ""
            return self._search_context
        self._search_context = ""

        queries = await self._decide_search_queries()
        logger.info("Planner will try search queries: %s", queries)

        summary_lines: List[str] = []
        max_queries_to_use = 3
        max_summary_lines = 8

        for q in queries:
            results = await self._fetch_search_results(q, max_results=8)
            if results:
                formatted = self._format_search_results(results)
                logger.info("Planner search results for query=%r:\n%s", q, formatted)
                # Collect up to two concise lines per query for a compact summary without sources
                for idx, line in enumerate(formatted.splitlines()):
                    if idx >= 2:
                        break
                    clean_line = self._strip_source(line)
                    summary_lines.append(f"{q[:50]}... -> {clean_line}")
                    if len(summary_lines) >= max_summary_lines:
                        break
                if len(summary_lines) >= max_summary_lines or len(summary_lines) >= max_queries_to_use * 2:
                    break
            else:
                logger.info("Planner search produced no results for query=%r", q)

        if summary_lines:
            self._search_context = "Concise search summary (links removed):\n" + "\n".join(summary_lines)
            logger.info("Planner aggregated concise search summary from %d lines.", len(summary_lines))
        else:
            logger.info("Planner search produced no usable results; proceeding without external context.")

        return self._search_context

    async def edit_task(self) -> str:
        if not self.planner_llm:
            return
        controller = Controller()
        planner_prompt = PlannerPrompt(controller.registry.get_prompt_description())
        system_message = planner_prompt.get_system_message().content
        search_context = await self._get_search_context()
        search_block = ""
        if search_context:
            search_block = f"Readable DuckDuckGo findings selected by planner (summary only):\n{search_context}\n\n"
        content = f"""
                {system_message}
                {search_block}
                Use the search findings above to populate the "important search info" field in every step with concise, useful insights that support that step. Include a short summary of the most relevant search findings for the overall task if helpful.
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

    async def continue_edit_task(self, info_memory, task_summary) -> str:
        if not self.planner_llm:
            return
        controller = Controller()
        planner_prompt = PlannerPrompt(controller.registry.get_prompt_description())
        search_context = await self._get_search_context()
        search_block = ""
        if search_context:
            search_block = f"Readable DuckDuckGo findings selected by planner (summary only):\n{search_context}\n\n"
        content = f"The summary of previous tasks are as follows: {task_summary}\n\n"
        content += f"The information memory for previous tasks are as follows: {info_memory}\n\n"
        content += f"The previous task you planned and being completed is as follows: '{self.plan_list}'.\n\n"
        content += search_block
        content += 'Use the search findings above to populate the "important search info" field in every step with concise, useful insights that support that step. Include a short summary of the most relevant search findings for the overall task if helpful.\n'
        content += f"Based on the above information memory and task summaries, please continue to edit and provide a detailed step-by-step plan for the overall task: '{self.task}'. Ensure that the plan is clear, actionable, avoid the previous plan you generated, and follows the required format."
        response = await self.planner_llm.ainvoke([planner_prompt.get_system_message(), HumanMessage(content=content)])
        reply_text = response.content.strip()
        reply_norm = reply_text.upper()
        if "REFUSE TO MAKE PLAN" in reply_norm:
            logging.error("Planner refused. Aborting.")
            raise SystemExit(1)
        return response.content
