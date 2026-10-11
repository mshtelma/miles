"""OfficeQA tools for the GLM-5.3-Flash agent loop.

``run_tool`` is the agent's ``MILES_TOOL_EXECUTOR`` (see ``examples/air_officeqa/agent.py``); the
reward is ``path_reward.score``. Run the agent with ``terminal_tools=["submit_report"]`` so the report
call ends the trajectory.

Every delivered result starts with ``[observation obs_N]``, numbered per trajectory in execution order,
so a report can cite it. The tools read the corpus of the
dataset a prompt names (``metadata.tool_context.dataset``) from ``$OFFICEQA_DATA_ROOT/<dataset>``,
which prepare_data.py writes and the launcher points at.
"""

import os
import re
import threading
from pathlib import Path

from examples.air_officeqa import corpus, sandbox
from miles.rollout.generate_utils.agent_errors import ToolCallError

SUBMIT_TOOL = "submit_report"
OBSERVATION_MARKER = "[observation {oid}]"
# Exact numeric match; path_reward.py scores the report with it.
ANSWER_TOLERANCE = 0.0
DEFAULT_DATA_ROOT = "/root/datasets"

_DATASET_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")
# tool name -> (function, required arguments, optional arguments)
_TOOLS = {
    "search_documents": (corpus.search_documents, {"query"}, {"top_k"}),
    "grep_documents": (corpus.grep_documents, {"pattern"}, {"file_name", "year", "regex"}),
    "read_document": (corpus.read_document, {"file_name"}, {"start_line", "num_lines"}),
    "list_documents": (corpus.list_documents, set(), {"year"}),
}

# The executor is a plain function hook, so the per-trajectory observation counter
# lives here; one int per trajectory for the life of the rollout process.
_observation_lock = threading.Lock()
_observation_counts: dict[str, int] = {}


def run_tool(name: str, arguments: dict, context: dict) -> str:
    text = _execute(name, arguments, task=context["task"])
    oid = _next_observation_id(context["session_ref"])
    return OBSERVATION_MARKER.format(oid=oid) + "\n" + text


def _execute(name: str, arguments: dict, *, task: dict) -> str:
    if name == "compute":
        _check_arguments(name, arguments, required={"code"}, optional=set())
        return sandbox.run_code(arguments["code"])
    if name not in _TOOLS:
        raise ToolCallError(f"{name} is not an executable tool")
    function, required, optional = _TOOLS[name]
    _check_arguments(name, arguments, required=required, optional=optional)
    return function(corpus.load_corpus(_data_dir(task)), **arguments)


def _check_arguments(name: str, arguments: dict, *, required: set[str], optional: set[str]) -> None:
    if missing := required - arguments.keys():
        raise ToolCallError(f"{name} is missing arguments: {sorted(missing)}")
    if unknown := arguments.keys() - required - optional:
        raise ToolCallError(f"{name} got unknown arguments: {sorted(unknown)}")


def _data_dir(task: dict) -> Path:
    dataset = task.get("dataset")
    if not isinstance(dataset, str) or not _DATASET_NAME.fullmatch(dataset):
        raise ValueError("OfficeQA task context must name its dataset")
    return Path(os.environ.get("OFFICEQA_DATA_ROOT", DEFAULT_DATA_ROOT)) / dataset


def _next_observation_id(session_ref: str) -> str:
    with _observation_lock:
        count = _observation_counts.get(session_ref, 0) + 1
        _observation_counts[session_ref] = count
    return f"obs_{count}"
