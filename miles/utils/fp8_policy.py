"""Checkpoint-driven FP8 exclusions shared by the Megatron export paths."""

from dataclasses import dataclass


@dataclass(frozen=True)
class FP8SkipPolicy:
    """Dotted module paths compiled once for all weights in an export iterator."""

    entries: tuple[str, ...]
    paths: frozenset[tuple[str, ...]]

    def keeps(self, hf_name: str) -> bool:
        parts = tuple(hf_name.removesuffix(".weight").split("."))
        return any(
            parts[start:end] in self.paths for start in range(len(parts)) for end in range(start + 1, len(parts) + 1)
        )


def build_fp8_skip_policy(quantization_config: dict, *, require_exclusions: bool = False) -> FP8SkipPolicy:
    entries = quantization_config.get("modules_to_not_convert")
    if not entries:
        if require_exclusions:
            raise ValueError("GLM-5.3 FP8 export requires checkpoint modules_to_not_convert metadata")
        entries = ()
    if not isinstance(entries, (list, tuple)):
        raise ValueError("FP8 modules_to_not_convert must be a list of module paths")
    normalized = _normalized_skip_entries(entries)
    return FP8SkipPolicy(entries=normalized, paths=frozenset(tuple(entry.split(".")) for entry in normalized))


def _normalized_skip_entries(entries: list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """Use the same ``model.`` prefix aliases as the pinned SGLang FP8 loader."""
    normalized = []
    for entry in entries:
        if not isinstance(entry, str) or not entry.rstrip("."):
            raise ValueError("FP8 exclusion entries must be nonempty module paths")
        base = entry.removeprefix("model.").rstrip(".")
        if base and any(not part for part in base.split(".")):
            raise ValueError(f"Invalid FP8 exclusion path: {entry!r}")
        normalized.extend(candidate for candidate in (base, f"model.{base}".rstrip(".")) if candidate)
    return tuple(dict.fromkeys(normalized))
