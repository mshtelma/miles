"""The OfficeQA agent's tools, scorer and limits, shared by training and the standalone eval.

scripts/run_air_officeqa.py passes ``agent_env`` to the Ray workers and eval/entry.py sets it for its
own process, so an eval episode runs exactly like a training one.
"""

import json

# 128 turns, the report tool ends an episode, 16K tokens kept free for the final answer, and 20 minutes
# for the judge.
AGENT_SETTINGS = {
    "max_turns": 128,
    "terminal_tools": ["submit_report"],
    "reserve_tokens": 16384,
    "score_timeout_s": 1200,
}
TOOL_EXECUTOR = "examples.air_officeqa.adapters:run_tool"
REWARD_FN = "examples.air_officeqa.path_reward:score"


def agent_env(data_root: str) -> dict[str, str]:
    """Environment of the processes that run the agent, for prompts prepared under ``data_root``."""
    return {
        "MILES_TOOL_EXECUTOR": TOOL_EXECUTOR,
        "MILES_REWARD_FN": REWARD_FN,
        "MILES_AGENT_SETTINGS": json.dumps(AGENT_SETTINGS, separators=(",", ":")),
        "OFFICEQA_DATA_ROOT": data_root,
    }
