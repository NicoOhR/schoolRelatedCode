import os
from dataclasses import dataclass

import dotenv
import pyspark
import spacy
from kafka import KafkaConsumer, KafkaProducer
from pyspark.sql import DataFrame
from pyspark.sql.functions import explode, udf
from pyspark.sql.session import SparkSession
from pyspark.sql.types import ArrayType, StringType, StructField


@udf(returnType=ArrayType(StringType()))
def extract_ent(x):
    # load needs to be moved into global for each executor
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(x)
    return [ent.orth_ for ent in doc.ents]


@dataclass
class Processor:
    spark: SparkSession
    articles: DataFrame
    producer: KafkaProducer
    stream: None

    def __init__(self):
        dotenv.load_dotenv()
        kafka_server = os.environ["KAFKA_SERVER"]
        self.spark = (
            SparkSession.builder.master("local[*]")
            .config(
                "spark.jars.packages",
                "org.apache.spark:spark-sql-kafka-0-10_2.13:4.3.0",
            )
            .appName("processor")
            .getOrCreate()
        )
        self.stream = (
            self.spark.readStream.format("kafka")
            .option("kafka.bootstrap.servers", kafka_server)
            .option("subscribe", "articles")
            .load()
            .withColumn(
                "entities",
                extract_ent(pyspark.sql.functions.col("value").cast("string")),
            )
        )

    def get_stream(self):
        return (
            self.stream.select(explode(self.stream.entities).alias("entities"))
            .groupBy("entities")
            .count()
        )
