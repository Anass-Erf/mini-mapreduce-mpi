# Mini MapReduce Engine with MPI

## Project Description

This university project implements a simple MapReduce engine in Python using MPI. The engine solves the classic Word Count problem, then compares its result and execution time with an Apache Spark implementation. A Streamlit web interface is included to make the project easy to demonstrate.

The goal is educational: the MPI version shows how Map, Shuffle, and Reduce can be built manually, while Spark shows how the same problem is solved with a high-level distributed data framework.

## Objectives

- Implement a simplified MapReduce workflow with MPI.
- Use `mpi4py` to distribute work between processes.
- Implement Word Count with Map, Shuffle, Reduce, and Gather phases.
- Implement the same Word Count problem using Apache Spark.
- Benchmark MPI with 2, 4, and 6 processes against Spark local mode.
- Present results using a Streamlit web interface.

## Technologies Used

- Python 3
- mpi4py
- Open MPI
- Apache Spark with PySpark
- Streamlit
- pandas
- matplotlib

## Project Architecture

```text
mini-mapreduce-mpi-web/
├── data/
│   ├── input.txt
│   └── sample_large.txt
├── src/
│   ├── mini_mapreduce.py
│   ├── spark_wordcount.py
│   ├── benchmark.py
│   └── utils.py
├── web/
│   └── app.py
├── results/
│   ├── mpi_output.txt
│   ├── spark_output.txt
│   └── comparison.csv
├── docs/
│   └── project_explanation.md
├── requirements.txt
├── README.md
└── .gitignore
```

## Map, Shuffle, Reduce

### Map Phase

Rank 0 reads the input file and splits the lines between MPI processes using `comm.scatter`. Each process receives a chunk of lines and counts words locally.

Example local result:

```text
mpi 3
mapreduce 2
data 1
```

### Shuffle Phase

Each process sends every word to a reducer process using a deterministic hash:

```python
destination_process = stable_word_hash(word) % number_of_processes
```

This guarantees that the same word always goes to the same reducer process. Communication is done with MPI `alltoall`.

### Reduce Phase

Each reducer process aggregates all counts it received for its assigned words. Rank 0 gathers the reduced dictionaries using `comm.gather` and writes the final sorted result.

## Installation on Ubuntu

Install system dependencies:

```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv openmpi-bin libopenmpi-dev openjdk-17-jdk -y
```

Create and activate a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install Python dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

## Run the MPI Version

From the project root:

```bash
mpirun -np 4 python3 src/mini_mapreduce.py data/input.txt results/mpi_output.txt
```

Output file:

```text
results/mpi_output.txt
```

## Run the Spark Version

```bash
python3 src/spark_wordcount.py data/input.txt results/spark_output.txt
```

Output file:

```text
results/spark_output.txt
```

## Run the Benchmark

```bash
python3 src/benchmark.py
```

The benchmark compares:

- MPI with 2 processes
- MPI with 4 processes
- MPI with 6 processes
- Spark in local mode

Results are saved to:

```text
results/comparison.csv
```

## Run the Streamlit Web Interface

```bash
streamlit run web/app.py
```

The web app allows you to:

- Upload a text file.
- Select the number of MPI processes.
- Run MPI MapReduce.
- Run Apache Spark.
- Run the full benchmark.
- View word count tables and charts.
- Compare MPI and Spark execution times.

## Example Output

```text
mapreduce 4
mpi 4
phase 3
spark 3
word 3
```

Each line contains:

```text
word count
```

## Comparison Between MPI and Spark

MPI gives direct control over process communication. It is useful for understanding distributed algorithms because the programmer explicitly defines how data is split, exchanged, and reduced.

Spark provides a high-level API for distributed data processing. It automatically manages scheduling, partitioning, and fault tolerance, which makes it more practical for large production workloads.

For this small project, MPI may be faster for simple input because it has less framework overhead. Spark becomes more interesting when the data is larger and the processing pipeline is more complex.

## What to Present to the Professor

1. Explain the Word Count problem.
2. Show the MPI code and identify Map, Shuffle, Reduce, and Gather.
3. Run the MPI command with several process counts.
4. Run the Spark version on the same input.
5. Run the benchmark and open `results/comparison.csv`.
6. Open the Streamlit interface and demonstrate upload, execution, tables, and charts.
7. Explain that MPI is low-level and educational, while Spark is high-level and production-oriented.
