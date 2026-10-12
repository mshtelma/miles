"""The eval entry point: eval-set paths, prompt token counting and its light imports."""

import asyncio
import faulthandler
import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from examples.air_officeqa.eval import entry
from examples.air_officeqa.models import MODELS


def test_eval_set_paths_are_relative_to_staged_inputs(tmp_path):
    assert entry._eval_sets(["heldout=data/officeqa/eval.jsonl"], tmp_path) == {
        "heldout": tmp_path / "data/officeqa/eval.jsonl"
    }
    with pytest.raises(ValueError, match="unique"):
        entry._eval_sets(["x=/tmp/a", "x=/tmp/b"], tmp_path)


def test_the_token_counter_renders_a_prior_tool_call_the_way_sglang_does(tmp_path) -> None:
    # The agent sends tool-call arguments as a JSON string, and the GLM template iterates them as a dict
    # (chat_template.jinja: ``tc.arguments.items()``).
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    words.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=words, unk_token="[UNK]")
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m.role }} {{ m.content }} {% for tc in m.tool_calls or [] %}"
        "{% for k, v in tc.function.arguments.items() %}{{ k }} {{ v }} {% endfor %}{% endfor %}{% endfor %}"
    )
    tokenizer.save_pretrained(tmp_path)
    messages = [
        {"role": "user", "content": "q"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "search", "arguments": '{"query": "x"}'}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "obs"},
    ]

    count = entry._token_counter(entry._tokenizer(tmp_path), MODELS["GLM-5.3-Flash"])

    assert count(messages, None) == len("user q assistant query x tool obs".split())


def test_stop_texts_are_the_checkpoints_eos_tokens(tmp_path) -> None:
    class Tokenizer:
        def convert_ids_to_tokens(self, ids):
            return [{248046: "<|im_end|>", 248044: "<|endoftext|>"}[token] for token in ids]

    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [248046, 248044]}))
    assert entry._stop_texts(tmp_path, Tokenizer()) == ("<|im_end|>", "<|endoftext|>")
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": 248046}))
    assert entry._stop_texts(tmp_path, Tokenizer()) == ("<|im_end|>",)


@pytest.mark.parametrize("model_name", sorted(MODELS))
def test_the_base_model_is_served_from_its_own_directory(tmp_path, model_name) -> None:
    assert entry.model_source(tmp_path, "base", model_name=model_name) == (
        tmp_path / "models" / model_name,
        {"kind": "base"},
    )


def _save(root: Path, step: int, *, copied: int, runs: dict | None = None) -> Path:
    save = root / f"iter_{step:07d}"
    (save / "hf").mkdir(parents=True)
    (save / "hf/.complete").write_text("")
    (root / "latest_checkpointed_iteration.txt").write_text(str(copied))
    if runs is not None:
        (root / entry.MLFLOW_RUNS).write_text(json.dumps(runs))
    return save


def test_a_save_is_served_from_its_hf_export_with_its_training_run(tmp_path):
    save = _save(tmp_path, 25, copied=50, runs={"25": "train-run"})

    path, model = entry.model_source(tmp_path, str(save), model_name="GLM-5.3-Flash")

    assert path == save / "hf"
    assert model == {"kind": "checkpoint", "path": str(save), "rollout_id": 25, "mlflow_run_id": "train-run"}


def test_a_save_the_copy_has_not_reached_is_refused(tmp_path):
    save = _save(tmp_path, 25, copied=0)

    with pytest.raises(ValueError, match="fully copied"):
        entry.model_source(tmp_path, str(save), model_name="GLM-5.3-Flash")


def test_the_eval_reads_the_file_the_publisher_writes():
    from tools.air import publisher

    assert entry.MLFLOW_RUNS == publisher.MLFLOW_RUNS


def test_eval_imports_without_training_or_model_dependencies():
    script = (
        "import sys\n"
        "for name in ('torch', 'ray', 'megatron', 'sglang', 'transformers'):\n"
        "    sys.modules[name] = None\n"
        "import examples.air_officeqa.eval.entry\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[5])


def test_a_stalled_eval_dumps_once_until_turns_finish_again(tmp_path, monkeypatch):
    dumps = []
    monkeypatch.setattr(entry, "_dump_stacks", lambda reason, sglang_log: dumps.append(reason))
    now = [0.0]
    watch = entry._StallWatch(tmp_path / "sglang.log", stall_s=600, clock=lambda: now[0])
    try:
        for now[0], turns in [(300, 1), (850, 1), (950, 1), (990, 1), (1000, 2), (1700, 2)]:
            watch.tick(turns)
    finally:
        faulthandler.cancel_dump_traceback_later()
    assert dumps == ["no model turn finished for 650 s", "no model turn finished for 700 s"]


def test_the_stall_dump_shows_where_tasks_wait_and_the_end_of_the_sglang_log(tmp_path, monkeypatch, caplog):
    threads = []
    monkeypatch.setattr(entry.faulthandler, "dump_traceback", lambda **kwargs: threads.append(kwargs))
    sglang_log = tmp_path / "sglang.log"
    sglang_log.write_text("".join(f"line {i}\n" for i in range(100)))

    async def waits_for_the_server(event):
        await event.wait()

    async def main():
        event = asyncio.Event()
        tasks = [asyncio.create_task(waits_for_the_server(event)) for _ in range(3)]
        await asyncio.sleep(0)
        entry._dump_stacks("stalled", sglang_log)
        event.set()
        await asyncio.gather(*tasks)

    with caplog.at_level(logging.ERROR):
        asyncio.run(main())
    assert threads == [{"all_threads": True}]
    assert "3 tasks wait at" in caplog.text and "waits_for_the_server > locks.py" in caplog.text
    assert "line 99" in caplog.text and "line 49\n" not in caplog.text
