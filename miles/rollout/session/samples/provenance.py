"""Names owned by the session collector rather than by an agent or dataset."""

COLLECTOR_METADATA_KEYS = frozenset(
    {
        "original_token_count",
        "retained_token_count",
        "removed_turn_count",
        "truncated_turn_count",
        "retained_terminal_record_id",
        "reward_valid_for_retained_tokens",
    }
)

SERVER_OWNED_METADATA_KEYS = COLLECTOR_METADATA_KEYS | {
    "accumulated_token_ids",
    "tito_session_mismatch",
    "leaf",
    "turn_args",
}


def untrusted_agent_metadata(value: dict | None) -> dict:
    """Keep agent supplied fields that cannot overwrite collector evidence."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError("agent metadata must be an object")
    return {key: item for key, item in value.items() if key not in SERVER_OWNED_METADATA_KEYS}
