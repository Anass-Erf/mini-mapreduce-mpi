"""Apache Spark word count for comparison with the MPI version.

Run:
    python3 src/spark_wordcount.py data/input.txt results/spark_output.txt
"""

from __future__ import annotations

import sys
import os
import re
import time
from pathlib import Path

from pyspark.sql import SparkSession

from utils import write_word_counts


WORD_RE = re.compile(r"[^a-z0-9]+")


def clean_spark_word(word: str) -> str:
    """Normalize words inside Spark workers."""
    return WORD_RE.sub("", word.lower())


def run_spark_wordcount(input_file: str, output_file: str) -> float:
    start_time = time.perf_counter()
    os.environ.setdefault("SPARK_LOCAL_IP", "127.0.0.1")
    os.environ.setdefault("SPARK_LOCAL_HOSTNAME", "localhost")

    spark = (
        SparkSession.builder.appName("MiniMapReduceSparkWordCount")
        .master("local[*]")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    try:
        lines = spark.sparkContext.textFile(input_file)
        counts = (
            lines.flatMap(lambda line: line.split())
            .map(clean_spark_word)
            .filter(lambda word: word != "")
            .map(lambda word: (word, 1))
            .reduceByKey(lambda left, right: left + right)
            .collect()
        )
    finally:
        spark.stop()

    elapsed_time = time.perf_counter() - start_time
    write_word_counts(dict(counts), output_file)
    print(f"Apache Spark finished in {elapsed_time:.6f} seconds")
    print(f"Output written to {output_file}")
    return elapsed_time


def main() -> None:
    if len(sys.argv) != 3:
        script = Path(sys.argv[0]).name
        print(f"Usage: python3 src/{script} data/input.txt results/spark_output.txt")
        sys.exit(1)

    run_spark_wordcount(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    main()
