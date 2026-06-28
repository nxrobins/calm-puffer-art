from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


DEFAULT_SIGIL_CORPUS_DIR = Path(r"D:\SIGIL\corpus\out")
DEFAULT_SIGIL_IDIOM_JSONL = DEFAULT_SIGIL_CORPUS_DIR / "idiom.jsonl"
DEFAULT_SIGIL_IMPLEMENTATION_JSONL = DEFAULT_SIGIL_CORPUS_DIR / "implementation.jsonl"
DEFAULT_SIGIL_EXE = Path(r"D:\SIGIL\target\release\sigil.exe")
SIGIL_VERIFY_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class SigilCorpus:
    prompts: tuple[str, ...]
    training_outputs: tuple[str, ...]
    idiom_rows: tuple[Mapping[str, Any], ...]
    implementation_rows: tuple[Mapping[str, Any], ...]

    @property
    def prompt_count(self) -> int:
        return len(self.prompts)

    @property
    def training_output_count(self) -> int:
        return len(self.training_outputs)


def verify_sigil_code(code: str) -> bool:
    """Return True when the local Sigil compiler accepts inline source."""

    sigil_exe = _sigil_exe_path()
    if not sigil_exe.exists():
        return False
    try:
        completed = subprocess.run(
            [str(sigil_exe), "check-inline", "--json", code],
            text=True,
            capture_output=True,
            timeout=SIGIL_VERIFY_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return False
    return payload.get("status") == "ok"


def load_sigil_corpus(
    *,
    idiom_path: str | Path = DEFAULT_SIGIL_IDIOM_JSONL,
    implementation_path: str | Path = DEFAULT_SIGIL_IMPLEMENTATION_JSONL,
) -> SigilCorpus:
    idiom_rows = _load_jsonl(Path(idiom_path))
    implementation_rows = _load_jsonl(Path(implementation_path))
    prompts = tuple(
        intent
        for row in idiom_rows
        if isinstance((intent := row.get("intent")), str) and intent.strip()
    )
    implementation_outputs = [
        output
        for row in implementation_rows
        if isinstance((output := row.get("output")), str) and output.strip()
    ]
    idiom_outputs = [
        f"module demo;\n{output.strip()}\n"
        for row in idiom_rows
        if isinstance((output := row.get("output")), str) and output.strip()
    ]
    training_outputs = tuple(implementation_outputs + idiom_outputs)
    if not prompts:
        raise ValueError("sigil_idiom_prompts_empty")
    if not training_outputs:
        raise ValueError("sigil_training_outputs_empty")
    return SigilCorpus(
        prompts=prompts,
        training_outputs=training_outputs,
        idiom_rows=idiom_rows,
        implementation_rows=implementation_rows,
    )


def _load_jsonl(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid_jsonl:{path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"jsonl_row_not_object:{path}:{line_number}")
            rows.append(row)
    return tuple(rows)


def _sigil_exe_path() -> Path:
    override = os.environ.get("SIGIL_EXE")
    return Path(override) if override else DEFAULT_SIGIL_EXE


__all__ = [
    "DEFAULT_SIGIL_CORPUS_DIR",
    "DEFAULT_SIGIL_EXE",
    "DEFAULT_SIGIL_IDIOM_JSONL",
    "DEFAULT_SIGIL_IMPLEMENTATION_JSONL",
    "SIGIL_VERIFY_TIMEOUT_S",
    "SigilCorpus",
    "load_sigil_corpus",
    "verify_sigil_code",
]
