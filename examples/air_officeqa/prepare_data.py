"""Build the GLM-5.3-Flash OfficeQA prompt files and corpus directory.

Inputs, from the ``databricks/officeqa`` dataset on Hugging Face at ``OFFICEQA_REVISION``:
  * ``officeqa_full.csv``: the benchmark, 113 easy and 133 hard questions;
  * ``treasury_bulletins_transformed.zip``: the Treasury Bulletins as text, which the tools search.

The dataset is gated: accept its terms on Hugging Face, then export ``HF_TOKEN``. ``--source DIR``
reads both files from DIR instead.

Training (``train.jsonl``) gets every easy question plus ``--hard-train`` hard ones; the other hard
questions are held out for evaluation in ``eval_heldout.jsonl``.

``--out-dir`` receives the two prompt files, ``curriculum.json`` for curriculum.py, the bulletins and
their search passages (``chunks.jsonl``, cut by ``corpus.build_chunks``), so the directory is everything a
run needs. Prompts name it by its directory name (``metadata.tool_context.dataset``), and the tools read
it from ``$OFFICEQA_DATA_ROOT/<name>``; the launcher sets ``OFFICEQA_DATA_ROOT`` to its ``--data-dir``.

Output: one Miles JSONL row per question::

    prompt    [system, user]; the user message is the question, unchanged
    label     gold answer
    metadata  tools, reference_answer (read only by the scorer), uid, question and
              question_requirements (read by the path-report judge), tool_context.dataset

    python -m examples.air_officeqa.prepare_data --out-dir /root/datasets/officeqa
"""

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from examples.air_officeqa.corpus import CHUNKS_FILE, CORPUS_ZIP, build_chunks
from huggingface_hub import hf_hub_download

OFFICEQA_REPO = "databricks/officeqa"
OFFICEQA_REVISION = "763a8366abf2a3605c381d53586d844dc60fa756"
QUESTIONS_FILE = "officeqa_full.csv"

# Sampling settings for curriculum.py; see its module docstring.
CURRICULUM_SETTINGS = {"seed": 0, "floor": 0.01, "decay": 0.75, "max_groups_per_question": 2}
TOOLS = json.loads(Path(__file__).with_name("tools.json").read_text(encoding="utf-8"))

SYSTEM_PROMPT = (
    "You are a meticulous financial-data research agent answering questions about U.S. "
    "Treasury Bulletins. Ground every answer in figures you actually retrieve.\n\n"
    "Tools (each result is tagged on its first line as [observation obs_N] -- an id you must "
    "cite later):\n"
    "- search_documents(query, top_k): find the file/table for a figure.\n"
    "- grep_documents(pattern, file_name, year, regex): pin an exact row label/value.\n"
    "- read_document(file_name, start_line, num_lines): read the table cells + headers.\n"
    "- list_documents(year): list available bulletins.\n"
    "- compute(code): run Python (numpy/pandas) for ALL arithmetic.\n"
    "- submit_report(answer, path): FINISH the task. This is the ONLY way to end.\n\n"
    "How this task runs -- plan your steps around it:\n"
    "- You have up to [[MAX_TURNS]] steps, but you need not use them all: call submit_report as "
    "soon as you are confident in your answer.\n"
    "- If you have retrieved figures relevant to the question, commit your BEST supported answer "
    'and cite the observations behind it. Reserve answer="DATA NOT AVAILABLE" for when nothing you '
    "retrieved is relevant -- do not decline merely because you are unsure.\n\n"
    "FINISH by calling submit_report with your answer and the supporting path as its arguments. "
    "Do NOT print the JSON as text -- call the tool. Its arguments:\n"
    '  answer: "<ONLY the final value, in the units the question asks -- no working or other '
    'figures>"\n'
    '  path: a list of {"id","observation":"obs_N you received","claim":"row/period/units/value '
    'or the calculation","depends_on":["earlier id",...]}\n'
    "Rules for the path: cite ONLY observation ids that were delivered to you; a compute step "
    "must list in depends_on the read/search observations whose figures it used; report the path "
    "that actually supports the answer, not every dead end."
)


@dataclass(frozen=True)
class Question:
    uid: str
    question: str
    answer: str
    requirements: str = ""


def official_split(rows: list[dict], *, hard_train: int) -> tuple[list[Question], list[Question]]:
    """(train, held out): every easy question plus ``hard_train`` hard ones, then the other hard ones.

    Hard questions are taken in the order of a hash of their uid, so the split is fixed
    without a seed and does not follow the benchmark's own ordering.
    """
    by_difficulty: dict[str, list[Question]] = {"easy": [], "hard": []}
    for row in sorted(rows, key=lambda row: row["uid"]):
        if row["difficulty"] not in by_difficulty:
            raise ValueError(f"{row['uid']}: unknown difficulty {row['difficulty']!r}")
        by_difficulty[row["difficulty"]].append(
            Question(uid=row["uid"], question=row["question"].strip(), answer=row["answer"].strip())
        )
    hard = sorted(by_difficulty["hard"], key=lambda question: hashlib.sha256(question.uid.encode()).hexdigest())
    if not 0 < hard_train < len(hard):
        raise ValueError("hard_train must leave hard questions in both training and the held-out set")
    return by_difficulty["easy"] + hard[:hard_train], hard[hard_train:]


def build_rows(questions: list[Question], *, max_turns: int, dataset: str) -> list[dict]:
    if max_turns <= 0:
        raise ValueError("max_turns must be positive")
    system = SYSTEM_PROMPT.replace("[[MAX_TURNS]]", str(max_turns))
    return [_row(question, system=system, dataset=dataset) for question in questions]


def _row(question: Question, *, system: str, dataset: str) -> dict:
    if not question.question.strip() or not question.answer:
        raise ValueError(f"{question.uid}: expected a question and a gold answer")
    return {
        "prompt": [{"role": "system", "content": system}, {"role": "user", "content": question.question}],
        "label": question.answer,
        "metadata": {
            "uid": question.uid,
            "question": question.question,
            "question_requirements": question.requirements,
            "reference_answer": question.answer,
            "tools": TOOLS,
            "tool_context": {"dataset": dataset},
        },
    }


def _write_jsonl(rows: list[dict], path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_files(source: Path | None) -> tuple[Path, Path]:
    """The questions CSV and the bulletins zip: from ``source``, or downloaded at the pinned revision."""
    if source is not None:
        return source / QUESTIONS_FILE, source / CORPUS_ZIP
    questions, bulletins = (
        hf_hub_download(OFFICEQA_REPO, path, repo_type="dataset", revision=OFFICEQA_REVISION)
        for path in (QUESTIONS_FILE, f"treasury_bulletins_parsed/transformed/{CORPUS_ZIP}")
    )
    return Path(questions), Path(bulletins)


def write_chunks(bulletins: Path, path: Path) -> int:
    count = 0
    with zipfile.ZipFile(bulletins) as archive, path.open("x", encoding="utf-8") as stream:
        for member in sorted(name for name in archive.namelist() if name.endswith(".txt")):
            for chunk in build_chunks(Path(member).name, archive.read(member).decode("utf-8", errors="ignore")):
                stream.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--source",
        type=Path,
        help=f"directory with {QUESTIONS_FILE} and {CORPUS_ZIP}; without it both are downloaded from Hugging Face",
    )
    parser.add_argument("--hard-train", type=int, default=90, help="hard questions to train on; the rest are held out")
    parser.add_argument("--max-turns", type=int, default=128, help="the agent's step budget, as the prompt states it")
    args = parser.parse_args()
    questions, bulletins = source_files(args.source)
    with questions.open(newline="", encoding="utf-8") as stream:
        train, held_out = official_split(list(csv.DictReader(stream)), hard_train=args.hard_train)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_questions in (("train", train), ("eval_heldout", held_out)):
        rows = build_rows(split_questions, max_turns=args.max_turns, dataset=args.out_dir.name)
        _write_jsonl(rows, args.out_dir / f"{split}.jsonl")
        print(f"{split}: {len(rows)} rows -> {args.out_dir / f'{split}.jsonl'}")
    with (args.out_dir / "curriculum.json").open("x", encoding="utf-8") as stream:
        json.dump(CURRICULUM_SETTINGS, stream, indent=1, sort_keys=True)
    shutil.copyfile(bulletins, args.out_dir / CORPUS_ZIP)
    print(f"{CHUNKS_FILE}: {write_chunks(bulletins, args.out_dir / CHUNKS_FILE)} passages")


if __name__ == "__main__":
    main()
