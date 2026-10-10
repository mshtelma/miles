"""OfficeQA task tests: corpus tools, observation ids, and the prepared prompt files."""

import csv
import json
import subprocess
import sys
import zipfile

import pytest

from examples.glm5_3_flash.officeqa import adapters, corpus
from miles.rollout.generate_utils.agent_errors import ToolCallError

_JUNE_1940 = "\n".join(
    ["TREASURY BULLETIN JUNE 1940", "Budget expenditures", "National defense   2,602", "Veterans   557"]
)


@pytest.fixture
def task(tmp_path, monkeypatch):
    data = tmp_path / "data" / "oqa-test"
    data.mkdir(parents=True)
    with zipfile.ZipFile(data / corpus.CORPUS_ZIP, "w") as archive:
        archive.writestr("treasury_bulletin_1940_06.txt", _JUNE_1940)
        archive.writestr("treasury_bulletin_1941_01.txt", "January 1941\nNational defense   6,404")
    chunk = {"content": _JUNE_1940, "source": "treasury_bulletin_1940_06.txt", "bulletin_date": "1940-06"}
    (data / corpus.CHUNKS_FILE).write_text(json.dumps(chunk) + "\n")
    monkeypatch.setenv("OFFICEQA_DATA_ROOT", str(tmp_path / "data"))
    return {"dataset": "oqa-test"}


def _context(task: dict, session_ref: str) -> dict:
    return {"task": task, "session_ref": session_ref, "attempt_id": "a", "tool_call_id": "t", "tool_deadline_s": 60.0}


def test_tool_results_carry_per_trajectory_observation_ids(task, tmp_path):
    first, second = f"{tmp_path}-first", f"{tmp_path}-second"

    grep = adapters.run_tool("grep_documents", {"pattern": "National defense", "year": "1940"}, _context(task, first))
    read = adapters.run_tool(
        "read_document",
        {"file_name": "treasury_bulletin_1940_06", "start_line": 2, "num_lines": 1},
        _context(task, first),
    )
    search = adapters.run_tool("search_documents", {"query": "national defense 1940"}, _context(task, first))
    listing = adapters.run_tool("list_documents", {}, _context(task, second))

    assert grep.startswith("[observation obs_1]\n")
    assert "treasury_bulletin_1940_06.txt:2: National defense   2,602" in grep
    assert read.startswith("[observation obs_2]\n") and "2: National defense   2,602" in read
    assert search.startswith("[observation obs_3]\n") and "treasury_bulletin_1940_06.txt (1940-06)" in search
    assert listing.startswith("[observation obs_1]\n") and "2 document(s)" in listing


@pytest.mark.parametrize(
    ("name", "arguments", "message"),
    [
        ("search_documents", {}, "missing arguments"),
        ("read_document", {"file_name": "x", "path": "/etc/passwd"}, "unknown arguments"),
        ("submit_report", {"answer": "1", "path": []}, "not an executable tool"),
    ],
)
def test_policy_argument_mistakes_are_tool_errors(task, tmp_path, name, arguments, message):
    with pytest.raises(ToolCallError, match=message):
        adapters.run_tool(name, arguments, _context(task, str(tmp_path)))


def test_compute_runs_through_the_sandbox(task, tmp_path, monkeypatch):
    monkeypatch.setattr(adapters.sandbox, "run_code", lambda code: f"sandboxed: {code}")

    result = adapters.run_tool("compute", {"code": "print(2602 + 557)"}, _context(task, str(tmp_path)))

    assert result == "[observation obs_1]\nsandboxed: print(2602 + 557)"


def test_prepared_rows_keep_the_question_and_drop_the_lockout_rules():
    from examples.glm5_3_flash.officeqa import prepare_data

    question = prepare_data.Question(uid="UID0001", question="What was it in 1940?", answer="2602")

    [row] = prepare_data.build_rows([question], max_turns=128, dataset="officeqa")

    system = row["prompt"][0]["content"]
    assert "up to 128 steps" in system and "[[" not in system and "are closed" not in system
    assert row["prompt"][1] == {"role": "user", "content": "What was it in 1940?"}
    assert row["label"] == row["metadata"]["reference_answer"] == "2602"
    assert row["metadata"]["question"] == "What was it in 1940?"
    assert row["metadata"]["tool_context"] == {"dataset": "officeqa"}
    assert [tool["function"]["name"] for tool in row["metadata"]["tools"]][-1] == "submit_report"


def test_official_split_trains_on_every_easy_question_and_holds_out_the_rest_of_the_hard():
    from examples.glm5_3_flash.officeqa import prepare_data

    rows = [{"uid": f"UID{i:04d}", "question": f"Q{i}?", "answer": str(i), "difficulty": "easy"} for i in range(3)]
    rows += [
        {"uid": f"UID{i:04d}", "question": f"Q{i}?", "answer": str(i), "difficulty": "hard"} for i in range(3, 10)
    ]

    train, held_out = prepare_data.official_split(rows, hard_train=4)
    again, _ = prepare_data.official_split(list(reversed(rows)), hard_train=4)

    assert [q.uid for q in train[:3]] == ["UID0000", "UID0001", "UID0002"]
    assert len(train) == 7 and len(held_out) == 3
    assert not {q.uid for q in train} & {q.uid for q in held_out}
    assert [q.uid for q in again] == [q.uid for q in train]  # fixed by uid, not by row order
    with pytest.raises(ValueError, match="both"):
        prepare_data.official_split(rows, hard_train=7)


def test_prepare_data_writes_a_self_contained_run_directory(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    with zipfile.ZipFile(corpus_dir / corpus.CORPUS_ZIP, "w") as archive:
        archive.writestr("treasury_bulletin_1940_06.txt", _JUNE_1940)
    (corpus_dir / corpus.CHUNKS_FILE).write_text("{}\n")
    questions = tmp_path / "officeqa_full.csv"
    with questions.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["uid", "question", "answer", "difficulty", "source_files"])
        writer.writeheader()
        for index in range(6):
            difficulty = "easy" if index < 2 else "hard"
            writer.writerow(
                {"uid": f"UID{index:04d}", "question": f"Q{index}?", "answer": str(index), "difficulty": difficulty}
            )
    out_dir = tmp_path / "datasets" / "officeqa"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "examples.glm5_3_flash.officeqa.prepare_data",
            "--questions",
            str(questions),
            "--corpus-dir",
            str(corpus_dir),
            "--out-dir",
            str(out_dir),
            "--hard-train",
            "2",
        ],
        check=True,
    )

    assert sorted(path.name for path in out_dir.iterdir()) == [
        corpus.CHUNKS_FILE,
        "curriculum.json",
        "eval_heldout.jsonl",
        "train.jsonl",
        corpus.CORPUS_ZIP,
    ]
    settings = json.loads((out_dir / "curriculum.json").read_text())
    assert settings == {"decay": 0.75, "floor": 0.01, "max_groups_per_question": 2, "seed": 0}
    rows = [json.loads(line) for line in (out_dir / "train.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {row["metadata"]["tool_context"]["dataset"] for row in rows} == {"officeqa"}
    assert len((out_dir / "eval_heldout.jsonl").read_text().splitlines()) == 2
