FROM apache/airflow:2.8.1

# root user to install compilers needed for duckdb
USER root
RUN apt-get update && apt-get install -y gcc g++

USER airflow
RUN pip install --no-cache-dir pandas duckdb requests python-dotenv