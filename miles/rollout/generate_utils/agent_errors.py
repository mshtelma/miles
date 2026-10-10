"""Shared failure classifications for agent adapters and rollout collection."""


class EnvironmentFailure(RuntimeError):
    """The sandbox, scorer, or model service failed independently of policy."""


class ToolCallError(ValueError):
    """A policy tool call was malformed or not allowed; return it to the policy."""
