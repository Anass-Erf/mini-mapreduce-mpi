"""Mini MapReduce word count implemented with MPI.

Run:
    mpirun -np 4 python3 src/mini_mapreduce.py data/input.txt results/mpi_output.txt
"""

from __future__ import annotations

import sys
import time
import hashlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from mpi4py import MPI

from utils import chunk_lines, read_text_file, tokenize_text, write_word_counts


def stable_word_hash(word: str) -> int:
    """Return the same hash value for a word in every MPI process."""
    digest = hashlib.md5(word.encode("utf-8")).hexdigest()
    return int(digest, 16)


def local_map(lines: List[str]) -> Dict[str, int]:
    """Map phase: each MPI process counts words in its own lines."""
    counter: Counter[str] = Counter()
    for line in lines:
        counter.update(tokenize_text(line))
    return dict(counter)


def build_shuffle_buckets(local_counts: Dict[str, int], size: int) -> List[Dict[str, int]]:
    """Shuffle phase: send each word to hash(word) % number_of_processes."""
    buckets: List[Dict[str, int]] = [defaultdict(int) for _ in range(size)]
    for word, count in local_counts.items():
        destination_process = stable_word_hash(word) % size
        buckets[destination_process][word] += count
    return [dict(bucket) for bucket in buckets]


def reduce_counts(received_buckets: List[Dict[str, int]]) -> Dict[str, int]:
    """Reduce phase: aggregate all counts received for this reducer process."""
    reduced: Counter[str] = Counter()
    for bucket in received_buckets:
        reduced.update(bucket)
    return dict(reduced)


def run_mpi_wordcount(input_file: str, output_file: str) -> float:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    start_time = MPI.Wtime()

    if rank == 0:
        text = read_text_file(input_file)
        lines = text.splitlines()
        chunks = chunk_lines(lines, size)
    else:
        chunks = None

    # Scatter phase: rank 0 distributes a chunk of lines to each process.
    local_lines = comm.scatter(chunks, root=0)

    # Map phase: each process creates local word counts.
    local_counts = local_map(local_lines)

    # Shuffle phase: all processes exchange buckets so identical words reach the same reducer.
    outgoing_buckets = build_shuffle_buckets(local_counts, size)
    incoming_buckets = comm.alltoall(outgoing_buckets)

    # Reduce phase: each process combines the counts it received.
    reduced_counts = reduce_counts(incoming_buckets)

    # Gather phase: rank 0 collects reduced dictionaries and writes the final result.
    gathered_counts = comm.gather(reduced_counts, root=0)

    elapsed_time = MPI.Wtime() - start_time
    if rank == 0:
        final_counts: Counter[str] = Counter()
        for partial_counts in gathered_counts:
            final_counts.update(partial_counts)
        write_word_counts(dict(final_counts), output_file)
        print(f"MPI MapReduce finished in {elapsed_time:.6f} seconds")
        print(f"Output written to {output_file}")

    return elapsed_time


def main() -> None:
    if len(sys.argv) != 3:
        if MPI.COMM_WORLD.Get_rank() == 0:
            script = Path(sys.argv[0]).name
            print(f"Usage: mpirun -np 4 python3 src/{script} data/input.txt results/mpi_output.txt")
        sys.exit(1)

    run_mpi_wordcount(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
