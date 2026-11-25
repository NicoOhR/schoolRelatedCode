import os
from glob import iglob
from itertools import chain

from graphframes import GraphFrame
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.12.3")
    .config("spark.driver.memory", "20g")
    .getOrCreate()
)

edges_raw = spark.read.csv(
    "./twitter/*.edges", sep=" ", schema="src LONG, dst LONG"
).withColumn(
    "ego",
    F.regexp_extract(F.col("_metadata.file_path"), r"(\d+)\.edges$", 1).cast("long"),
)

ego_edges = (
    edges_raw.select("ego", F.col("src").alias("dst"))
    .union(edges_raw.select("ego", F.col("dst")))
    .select(F.col("ego").alias("src"), "dst")
)

edges = ego_edges.select("src", "dst").union(edges_raw.select("src", "dst")).distinct()

vertices = (
    edges.select(F.col("src").alias("id"))
    .union(edges.select(F.col("dst").alias("id")))
    .distinct()
)

g = GraphFrame(vertices, edges)
g.outDegrees.sort("outDegree", ascending=False).show()
g.inDegrees.sort("inDegree", ascending=False).show()

g2 = g.pageRank(resetProbability=0.15, tol=0.01)
g2.vertices.sort("pagerank", ascending=False).show()

spark.sparkContext.setCheckpointDir("./tmp/graphframes_connected_componenets")
g.connectedComponents().groupBy("component").count().sort(
    "count", ascending=False
).show()

g.triangleCount().select("id", "count").sort("count", ascending=False).show()
