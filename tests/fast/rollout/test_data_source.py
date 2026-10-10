from pathlib import Path
from types import SimpleNamespace

from miles.rollout.data_source import RolloutDataSource
from miles.utils.types import Sample


def _make_args(**overrides) -> SimpleNamespace:
    defaults = dict(rollout_global_dataset=False, save=None, load=None, rollout_shuffle=False)
    return SimpleNamespace(**{**defaults, **overrides})


def test_save_writes_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The built-in source guards itself, so the executor needs no outer guard to keep it silent."""
    source = RolloutDataSource(_make_args(save=str(tmp_path)))

    source.save(rollout_id=3)

    assert list(tmp_path.iterdir()) == []


def test_load_reads_nothing_without_a_global_dataset(tmp_path: Path) -> None:
    """The load side has always been called unconditionally and relies on the same internal guard."""
    source = RolloutDataSource(_make_args(load=str(tmp_path)))

    source.load(rollout_id=3)

    assert source.sample_offset == 0
    assert source.epoch_id == 0


def _tiny_source(monkeypatch, size: int) -> RolloutDataSource:
    class TinyDataset:
        def __init__(self, *args, **kwargs):
            self.original = [Sample(prompt=f"prompt-{index}") for index in range(size)]
            self.samples = list(self.original)

        def __len__(self):
            return len(self.samples)

        def shuffle(self, epoch):
            shift = epoch % len(self.original)
            self.samples = self.original[shift:] + self.original[:shift]

    monkeypatch.setattr("miles.rollout.data_source.load_tokenizer", lambda *args, **kwargs: object())
    monkeypatch.setattr("miles.rollout.data_source.load_processor", lambda *args, **kwargs: object())
    monkeypatch.setattr("miles.rollout.data_source.Dataset", TinyDataset)
    args = _make_args(
        rollout_global_dataset=True,
        rollout_shuffle=True,
        hf_checkpoint="unused",
        chat_template_path=None,
        dump_details=None,
        prompt_data="prompts.jsonl",
        rollout_max_prompt_len=32,
        input_key="prompt",
        multimodal_keys=None,
        label_key=None,
        metadata_key="metadata",
        tool_key=None,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        rollout_seed=7,
        n_samples_per_prompt=2,
    )
    return RolloutDataSource(args)


def test_a_request_larger_than_the_dataset_wraps_through_several_epochs(monkeypatch) -> None:
    """An epoch boundary reshuffles, and a request may cross more than one of them."""
    source = _tiny_source(monkeypatch, size=2)

    groups = source.get_samples(5)

    assert [group[0].prompt for group in groups] == ["prompt-0", "prompt-1", "prompt-1", "prompt-0", "prompt-0"]
    assert (source.epoch_id, source.sample_offset) == (2, 1)
    assert all(len(group) == 2 for group in groups)


def test_a_request_for_no_prompts_leaves_the_cursor_alone(monkeypatch) -> None:
    source = _tiny_source(monkeypatch, size=2)

    assert source.get_samples(0) == []
    assert (source.epoch_id, source.sample_offset) == (0, 0)


def test_resume_restores_cursor_and_metadata_while_fresh_start_needs_no_cursor(tmp_path: Path, monkeypatch) -> None:
    source = _tiny_source(monkeypatch, size=2)
    source.args.load = str(tmp_path)
    source.load(rollout_id=-1)
    assert (source.epoch_id, source.sample_offset, source.metadata) == (0, 0, {})
    source.get_samples(3)
    source.metadata["curriculum"] = {"phase": "hard", "solved": ["q1"]}
    source.args.save = str(tmp_path)
    source.save(rollout_id=3)

    resumed = _tiny_source(monkeypatch, size=2)
    resumed.args.load = str(tmp_path)
    resumed.load(rollout_id=3)
    assert (resumed.epoch_id, resumed.sample_offset, resumed.sample_index) == (1, 1, 6)
    assert resumed.sample_group_index == 3
    assert resumed.metadata == source.metadata
    assert resumed.get_samples(1)[0][0].prompt == source.get_samples(1)[0][0].prompt
