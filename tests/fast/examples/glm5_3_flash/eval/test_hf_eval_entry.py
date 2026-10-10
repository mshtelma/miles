"""The eval entry point: eval-set paths, prompt token counting and its light imports."""

import subprocess
import sys
from pathlib import Path

import pytest

from examples.glm5_3_flash.officeqa.eval import entry


def test_eval_set_paths_are_relative_to_staged_inputs(tmp_path):
    assert entry._eval_sets(["heldout=data/officeqa/eval.jsonl"], tmp_path) == {
        "heldout": tmp_path / "data/officeqa/eval.jsonl"
    }
    with pytest.raises(ValueError, match="unique"):
        entry._eval_sets(["x=/tmp/a", "x=/tmp/b"], tmp_path)


def test_the_token_counter_renders_a_prior_tool_call_the_way_sglang_does(tmp_path) -> None:
    # An eval once lost every trajectory here: the agent sends tool-call arguments as a JSON string, and the GLM
    # template iterates them as a dict (chat_template.jinja: ``tc.arguments.items()``).
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

    count = entry._token_counter(tmp_path)

    assert count(messages, None) == len("user q assistant query x tool obs".split())


def test_eval_imports_without_training_or_model_dependencies():
    script = (
        "import sys\n"
        "for name in ('torch', 'ray', 'megatron', 'sglang', 'transformers'):\n"
        "    sys.modules[name] = None\n"
        "import examples.glm5_3_flash.officeqa.eval.entry\n"
    )
    subprocess.run([sys.executable, "-c", script], check=True, cwd=Path(__file__).resolve().parents[5])
