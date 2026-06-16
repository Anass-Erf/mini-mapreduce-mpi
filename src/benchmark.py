"""Benchmark MPI MapReduce and Apache Spark word count implementations."""

from __future__ import annotations

import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_FILE = PROJECT_ROOT / "data" / "sample_large.txt"
RESULTS_DIR = PROJECT_ROOT / "results"
COMPARISON_FILE = RESULTS_DIR / "comparison.csv"


def run_command(command: List[str], cwd: Path) -> float:
    start_time = time.perf_counter()
    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    elapsed_time = time.perf_counter() - start_time

    if completed.returncode != 0:
        print("Command failed:")
        print(" ".join(command))
        print(completed.stdout)
        print(completed.stderr, file=sys.stderr)
        raise RuntimeError(f"Command exited with code {completed.returncode}")

    return elapsed_time


def write_comparison(rows: List[Dict[str, str]]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with COMPARISON_FILE.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["engine", "processes", "input_file", "execution_time_seconds"],
        )
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows: List[Dict[str, str]]) -> None:
    print("\nBenchmark comparison")
    print("-" * 74)
    print(f"{'Engine':<16} {'Processes':<10} {'Input file':<24} {'Time (s)':>10}")
    print("-" * 74)
    for row in rows:
        print(
            f"{row['engine']:<16} {row['processes']:<10} "
            f"{Path(row['input_file']).name:<24} {row['execution_time_seconds']:>10}"
        )
    print("-" * 74)
    print(f"Saved to {COMPARISON_FILE}")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, str]] = []

    for processes in [2, 4, 6]:
        output_file = RESULTS_DIR / f"mpi_output_{processes}.txt"
        command = [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(processes),
            sys.executable,
            "src/mini_mapreduce.py",
            str(INPUT_FILE),
            str(output_file),
        ]
        elapsed = run_command(command, PROJECT_ROOT)
        rows.append(
            {
                "engine": "MPI",
                "processes": str(processes),
                "input_file": str(INPUT_FILE),
                "execution_time_seconds": f"{elapsed:.6f}",
            }
        )

    spark_output = RESULTS_DIR / "spark_output.txt"
    spark_command = [
        sys.executable,
        "src/spark_wordcount.py",
        str(INPUT_FILE),
        str(spark_output),
    ]
    elapsed = run_command(spark_command, PROJECT_ROOT)
    rows.append(
        {
            "engine": "Spark",
            "processes": "local[*]",
            "input_file": str(INPUT_FILE),
            "execution_time_seconds": f"{elapsed:.6f}",
        }
    )

    write_comparison(rows)
    print_table(rows)


if __name__ == "__main__":
    main()
