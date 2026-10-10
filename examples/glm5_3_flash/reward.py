"""The task reward of a GLM-5.3-Flash agent trajectory (the launcher's ``--custom-rm-path``)."""

import math

from miles.utils.types import Sample


async def reward_func(args, samples, **kwargs):
    """Score each retained trajectory; never reuse a discarded final answer's reward."""
    if isinstance(samples, list):
        return [_task_reward(sample) for sample in samples]
    return _task_reward(samples)


def _task_reward(sample: Sample) -> float:
    metadata = sample.metadata
    if (
        not isinstance(metadata, dict)
        or "reward" not in metadata
        or "reward_valid_for_retained_tokens" not in metadata
    ):
        raise ValueError("missing score or collector reward/retained-token attestation")
    recorded_reward = float(metadata["reward"])
    if not math.isfinite(recorded_reward):
        raise ValueError("non-finite task reward")
    valid = metadata["reward_valid_for_retained_tokens"]
    if type(valid) is not bool:
        raise ValueError("reward attestation must be boolean")
    task_reward = recorded_reward if valid else 0.0
    metadata["raw_reward"] = task_reward
    return task_reward
