"""Shared helpers for the Mini MapReduce MPI project."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List


WORD_RE = re.compile(r"[^a-z0-9]+")


def clean_word(word: str) -> str:
    """Return a lowercase word without punctuation."""
    return WORD_RE.sub("", word.lower())


def tokenize_text(text: str) -> List[str]:
    """Split text into normalized words."""
    words = []
    for raw_word in text.split():
        word = clean_word(raw_word)
        if word:
            words.append(word)
    return words


def read_text_file(input_file: str | Path) -> str:
    """Read a UTF-8 text file."""
    return Path(input_file).read_text(encoding="utf-8")


def write_word_counts(word_counts: Dict[str, int], output_file: str | Path) -> None:
    """Write sorted word counts using the format: word count."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sorted_items = sorted(word_counts.items(), key=lambda item: (-item[1], item[0]))
    lines = [f"{word} {count}" for word, count in sorted_items]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def load_word_counts(output_file: str | Path) -> Dict[str, int]:
    """Load a word-count result file written as: word count."""
    counts: Dict[str, int] = {}
    path = Path(output_file)
    if not path.exists():
        return counts

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        word, count = line.rsplit(maxsplit=1)
        counts[word] = int(count)
    return counts


def generate_sample_text(repetitions: int = 200) -> str:
    """Generate repeated sample text for benchmark demonstrations."""
    paragraph = (
        "Mini MapReduce with MPI demonstrates map shuffle reduce word count. "
        "Apache Spark provides a high level comparison for distributed data processing. "
    )
    return paragraph * repetitions


def chunk_lines(lines: Iterable[str], number_of_chunks: int) -> List[List[str]]:
    """Split lines into balanced chunks."""
    chunks: List[List[str]] = [[] for _ in range(number_of_chunks)]
    for index, line in enumerate(lines):
        chunks[index % number_of_chunks].append(line)
    return chunks
