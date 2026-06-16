"""Streamlit interface for the Mini MapReduce Engine with MPI."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
UPLOADED_FILE = DATA_DIR / "uploaded_input.txt"
MPI_OUTPUT = RESULTS_DIR / "mpi_output.txt"
SPARK_OUTPUT = RESULTS_DIR / "spark_output.txt"
COMPARISON_FILE = RESULTS_DIR / "comparison.csv"


def run_command(command: list[str]) -> tuple[bool, float, str, str]:
    start_time = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    elapsed = time.perf_counter() - start_time
    return completed.returncode == 0, elapsed, completed.stdout, completed.stderr


def load_counts(path: Path) -> pd.DataFrame:
    rows = []
    if not path.exists():
        return pd.DataFrame(columns=["word", "count"])

    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        word, count = line.rsplit(maxsplit=1)
        rows.append({"word": word, "count": int(count)})

    return pd.DataFrame(rows).sort_values(["count", "word"], ascending=[False, True])


def show_command_result(success: bool, stdout: str, stderr: str) -> None:
    if success:
        st.success("Command finished successfully.")
        if stdout.strip():
            st.code(stdout.strip(), language="text")
    else:
        st.error("Command failed.")
        if stdout.strip():
            st.code(stdout.strip(), language="text")
        if stderr.strip():
            st.code(stderr.strip(), language="text")


def show_word_results(title: str, output_file: Path, execution_time: float | None) -> None:
    st.subheader(title)
    if execution_time is not None:
        st.metric("Execution time", f"{execution_time:.4f} s")

    counts = load_counts(output_file)
    if counts.empty:
        st.info("No result file available yet.")
        return

    st.dataframe(counts.head(20), use_container_width=True, hide_index=True)

    top_words = counts.head(10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(top_words["word"], top_words["count"], color="#2f7d6d")
    ax.set_xlabel("Word")
    ax.set_ylabel("Count")
    ax.tick_params(axis="x", rotation=35)
    st.pyplot(fig)


def show_comparison() -> None:
    st.subheader("Comparison")
    if not COMPARISON_FILE.exists():
        st.info("Run the full benchmark to create the comparison table.")
        return

    comparison = pd.read_csv(COMPARISON_FILE)
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    labels = comparison["engine"] + " " + comparison["processes"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, comparison["execution_time_seconds"], color="#4464ad")
    ax.set_ylabel("Execution time (seconds)")
    ax.tick_params(axis="x", rotation=25)
    st.pyplot(fig)


def main() -> None:
    st.set_page_config(page_title="Mini MapReduce Engine with MPI", layout="wide")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    st.title("Mini MapReduce Engine with MPI")
    st.write(
        "This project implements Map, Shuffle, and Reduce using MPI and compares it with Apache Spark."
    )

    uploaded_file = st.file_uploader("Upload a .txt file", type=["txt"])
    input_file = DATA_DIR / "input.txt"
    if uploaded_file is not None:
        UPLOADED_FILE.write_bytes(uploaded_file.getvalue())
        input_file = UPLOADED_FILE
        st.success(f"Using uploaded file: {uploaded_file.name}")
    else:
        st.info("Using the default sample file from data/input.txt.")

    mpi_processes = st.selectbox("Number of MPI processes", [2, 4, 6, 8], index=1)

    col1, col2, col3 = st.columns(3)

    if col1.button("Run MPI MapReduce", use_container_width=True):
        command = [
            "mpirun",
            "--oversubscribe",
            "-np",
            str(mpi_processes),
            sys.executable,
            "src/mini_mapreduce.py",
            str(input_file),
            str(MPI_OUTPUT),
        ]
        success, elapsed, stdout, stderr = run_command(command)
        st.session_state["mpi_time"] = elapsed if success else None
        show_command_result(success, stdout, stderr)

    if col2.button("Run Apache Spark", use_container_width=True):
        command = [
            sys.executable,
            "src/spark_wordcount.py",
            str(input_file),
            str(SPARK_OUTPUT),
        ]
        success, elapsed, stdout, stderr = run_command(command)
        st.session_state["spark_time"] = elapsed if success else None
        show_command_result(success, stdout, stderr)

    if col3.button("Run Full Benchmark", use_container_width=True):
        command = [sys.executable, "src/benchmark.py"]
        success, elapsed, stdout, stderr = run_command(command)
        show_command_result(success, stdout, stderr)

    left, right = st.columns(2)
    with left:
        show_word_results("MPI Word Count", MPI_OUTPUT, st.session_state.get("mpi_time"))
    with right:
        show_word_results("Spark Word Count", SPARK_OUTPUT, st.session_state.get("spark_time"))

    show_comparison()

    st.subheader("Educational explanation")
    st.markdown(
        """
        **Map phase:** each process reads its assigned lines and creates local `(word, count)` pairs.

        **Shuffle phase:** each word is sent to `stable_word_hash(word) % number_of_processes`, so the same word always reaches the same reducer.

        **Reduce phase:** each reducer aggregates all counts received for its words.

        **MPI MapReduce vs Apache Spark:** MPI exposes communication directly and is excellent for learning distributed algorithms. Spark provides a higher level API, automatic scheduling, and a larger ecosystem for real data processing.
        """
    )


if __name__ == "__main__":
    main()
