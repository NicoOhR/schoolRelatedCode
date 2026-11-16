FROM ghcr.io/astral-sh/uv:python3.14-trixie

# Spark runs on the JVM: PySpark 4.x needs Java 17 or 21
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=/app/.venv/bin/python \
    UV_NO_DEV=1

WORKDIR /app

# Install dependencies first so they're cached when only the source changes
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-install-project

# spaCy model used by the extract_ent UDF
RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# Pre-download the Spark Kafka connector into the ivy cache so startup doesn't hit Maven
RUN uv run --no-project python -c "\
from pyspark.sql import SparkSession; \
SparkSession.builder.master('local[1]') \
.config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.13:4.3.0') \
.getOrCreate().stop()"

# Copy the project and install it
COPY . /app
RUN uv sync --locked --inexact \
    && rm -f .env \
    && mv .env_docker .env

# --no-sync so uv run does not remove the spaCy model (it is not in uv.lock)
CMD ["uv", "run", "--no-sync", "processor"]
