"""Agentic retrieval: the model decides when to search, what to query,
and when it has enough context to answer."""

from __future__ import annotations

import re
from collections.abc import Callable

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

_SYSTEM_TEMPLATE = """\
You are a helpful assistant that answers questions using the tools below.

Available tools:
{tool_descriptions}

Use this exact format for each reasoning step:

Thought: <your reasoning about what to do next>
Action: <tool name>
Action Input: <input string for the tool>

After the tool returns, you will see:

Observation: <tool output>

Repeat Thought/Action/Action Input/Observation as many times as needed.
When you have enough information, respond with:

Thought: I now have enough information to answer.
Final Answer: <your answer>

Important:
- Always start with a Thought.
- Only call one tool per step.
- Do NOT make up information. If the tools do not return useful results,
  say so in your Final Answer.
"""

_STEP_TEMPLATE = """\
Thought: {thought}
Action: {action}
Action Input: {action_input}
Observation: {observation}
"""


# ---------------------------------------------------------------------------
# Tool / SearchTool
# ---------------------------------------------------------------------------

class Tool:
    """A tool the agent can invoke."""

    def __init__(self, name: str, description: str, fn: Callable[[str], str]):
        self.name = name
        self.description = description
        self.fn = fn

    def __call__(self, input_str: str) -> str:
        return self.fn(input_str)


class SearchTool(Tool):
    """Wraps a Retriever into a Tool."""

    def __init__(
        self,
        retriever,
        texts_by_id: dict,
        name: str = "search",
        description: str = "Search the knowledge base",
    ):
        self.retriever = retriever
        self.texts_by_id = texts_by_id

        def _search(query: str) -> str:
            hits = retriever.search(query)
            if not hits:
                return "No results found."
            parts = []
            for rank, (doc_id, score, _meta) in enumerate(hits, 1):
                text = texts_by_id.get(doc_id, "(text unavailable)")
                parts.append(f"[{rank}] (score {score:.4f}) {text}")
            return "\n".join(parts)

        super().__init__(name=name, description=description, fn=_search)


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

_THOUGHT_RE = re.compile(r"Thought:\s*(.+?)(?=\nAction:|\nFinal Answer:|\Z)", re.DOTALL)
_ACTION_RE = re.compile(r"Action:\s*(.+)")
_ACTION_INPUT_RE = re.compile(r"Action Input:\s*(.+)")
_FINAL_ANSWER_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse_llm_output(text: str) -> dict:
    """Return a dict with keys depending on what was found:
    - If final answer: {"final_answer": str, "thought": str}
    - If action:       {"thought": str, "action": str, "action_input": str}
    - Otherwise:       {"raw": str}
    """
    final = _FINAL_ANSWER_RE.search(text)
    if final:
        thought_m = _THOUGHT_RE.search(text)
        return {
            "final_answer": final.group(1).strip(),
            "thought": thought_m.group(1).strip() if thought_m else "",
        }

    thought_m = _THOUGHT_RE.search(text)
    action_m = _ACTION_RE.search(text)
    action_input_m = _ACTION_INPUT_RE.search(text)

    if action_m and action_input_m:
        return {
            "thought": thought_m.group(1).strip() if thought_m else "",
            "action": action_m.group(1).strip(),
            "action_input": action_input_m.group(1).strip(),
        }

    return {"raw": text}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

class AgentLoop:
    """Agentic retrieval: the model decides when to search, what to search
    for, and when it has enough context to answer."""

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        tools: list[Tool],
        max_steps: int = 5,
    ):
        self.generate_fn = generate_fn
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps

    # -- public API ----------------------------------------------------------

    def run(self, question: str, system_prompt: str | None = None) -> dict:
        """Run the agent loop on *question*.

        Returns
        -------
        dict with keys:
            answer      – the final answer string
            steps       – list of step dicts (thought/action/action_input/observation)
            tool_calls  – total number of tool invocations
        """
        tool_desc = "\n".join(
            f"- {name}: {t.description}" for name, t in self.tools.items()
        )
        system = (system_prompt or "") + "\n" + _SYSTEM_TEMPLATE.format(
            tool_descriptions=tool_desc
        )

        steps: list[dict] = []
        prompt = self._build_prompt(system, question, steps)

        for _ in range(self.max_steps):
            llm_output = self.generate_fn(prompt)
            parsed = _parse_llm_output(llm_output)

            # ---- Final answer ------------------------------------------------
            if "final_answer" in parsed:
                return {
                    "answer": parsed["final_answer"],
                    "steps": steps,
                    "tool_calls": len(steps),
                }

            # ---- Tool call ---------------------------------------------------
            if "action" in parsed:
                action = parsed["action"]
                action_input = parsed["action_input"]
                thought = parsed.get("thought", "")

                tool = self.tools.get(action)
                if tool is None:
                    observation = f"Error: unknown tool '{action}'. Available: {', '.join(self.tools)}"
                else:
                    observation = tool(action_input)

                step = {
                    "thought": thought,
                    "action": action,
                    "action_input": action_input,
                    "observation": observation,
                }
                steps.append(step)
                prompt = self._build_prompt(system, question, steps)
                continue

            # ---- Unparseable output -- treat as final answer -----------------
            return {
                "answer": parsed.get("raw", llm_output).strip(),
                "steps": steps,
                "tool_calls": len(steps),
            }

        # Max steps exhausted -- ask for a final answer one last time
        prompt += "\nYou have reached the maximum number of steps. Please provide your Final Answer now.\n"
        llm_output = self.generate_fn(prompt)
        parsed = _parse_llm_output(llm_output)
        answer = parsed.get("final_answer", parsed.get("raw", llm_output)).strip()
        return {
            "answer": answer,
            "steps": steps,
            "tool_calls": len(steps),
        }

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _build_prompt(system: str, question: str, steps: list[dict]) -> str:
        parts = [system, f"\nQuestion: {question}\n"]
        for s in steps:
            parts.append(
                _STEP_TEMPLATE.format(
                    thought=s["thought"],
                    action=s["action"],
                    action_input=s["action_input"],
                    observation=s["observation"],
                )
            )
        return "\n".join(parts)
