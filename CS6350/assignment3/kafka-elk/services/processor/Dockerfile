FROM ghcr.io/astral-sh/uv:python3.14-trixie

RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-21-jre-headless procps \
    && rm -rf /var/lib/apt/lists/*

ENV PYSPARK_PYTHON=/app/.venv/bin/python \
    UV_NO_DEV=1

WORKDIR /app

COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --locked --no-install-project

RUN uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

RUN uv run --no-project python -c "\
from pyspark.sql import SparkSession; \
SparkSession.builder.master('local[1]') \
.config('spark.jars.packages', 'org.apache.spark:spark-sql-kafka-0-10_2.13:4.3.0') \
.getOrCreate().stop()"

COPY . /app
RUN uv sync --locked --inexact \
    && rm -f .env \
    && mv .env_docker .env

CMD ["uv", "run", "--no-sync", "processor"]
