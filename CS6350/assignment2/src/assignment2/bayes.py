import pathlib
import subprocess

import numpy as np
import pyspark
from pyspark import RDD
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StopWordsRemover, StringIndexer, Tokenizer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import functions as F
from pyspark.sql.session import SparkSession

diplo_train_path = pathlib.Path("./train.jsonl")
diplo_test_path = pathlib.Path("./test.jsonl")
diplo_validation_path = pathlib.Path("./validation.jsonl")

if not diplo_train_path.is_file():
    subprocess.run(
        [
            "wget",
            "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/refs/heads/master/data/train.jsonl",
        ],
        check=True,
    )

if not diplo_test_path.is_file():
    subprocess.run(
        [
            "wget",
            "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/refs/heads/master/data/test.jsonl",
        ],
        check=True,
    )

if not diplo_validation_path.is_file():
    subprocess.run(
        [
            "wget",
            "https://raw.githubusercontent.com/DenisPeskoff/2020_acl_diplomacy/refs/heads/master/data/validation.jsonl",
        ],
        check=True,
    )


spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.12.3")
    .config("spark.driver.memory", "20g")
    .getOrCreate()
)
sc = spark.sparkContext

# (label, text)
diplo_df = (
    spark.read.json(str(diplo_train_path))
    .union(spark.read.json(str(diplo_test_path)))
    .union(spark.read.json(str(diplo_validation_path)))
)

diplo_df = diplo_df.select(diplo_df.messages, diplo_df.sender_labels)
diplo_df = diplo_df.select(
    F.explode(F.arrays_zip("messages", "sender_labels")).alias("m")
).select(
    F.col("m.messages").alias("messages"),
    F.col("m.sender_labels").alias("sender_labels"),
)


tokenizer = Tokenizer(inputCol="messages", outputCol="raw")
stopWordRemover = StopWordsRemover(inputCol="raw", outputCol="filtered")
pipeline = Pipeline(stages=[tokenizer, stopWordRemover])
filtered_df = pipeline.fit(diplo_df).transform(diplo_df)

training, test = filtered_df.randomSplit([0.8, 0.2])


class NaiveBayes:
    p_lie = 0
    N_lie = 0
    N_tot = 0
    w = 1
    word_class = None

    def fit(self, training_df, w=1):
        self.w = w
        rdd = training_df.rdd
        self.N_lie = rdd.filter(lambda x: x[0] == False).count()
        self.N_tot = rdd.count()
        self.p_lie = self.N_lie / self.N_tot
        N_lie = self.N_lie
        N_tot = self.N_tot
        # ((word, class), p_x_class)
        self.word_class = (
            rdd.flatMap(lambda x: [(y, x[0]) for y in x[1]])
            .map(lambda x: ((x[0], x[1]), 1))
            .reduceByKey(lambda x, y: x + y)
            .map(
                lambda x: (
                    x[0],
                    x[1] / N_lie if x[0][1] == False else x[1] / (N_tot - N_lie),
                )
            )
        )

    def predict(self, df):
        table = sc.broadcast(self.word_class.collectAsMap())
        w = self.w
        rdd = df.select("filtered").rdd
        prior_lie = self.N_lie / self.N_tot
        prior_true = 1 - prior_lie

        N_tot = self.N_tot
        N_lie = self.N_lie

        def p_weighted(word):
            p_lie = table.value.get((word, False), 0)
            p_true = table.value.get((word, True), 0)
            n_word = p_lie * N_lie + p_true * (N_tot - N_lie)
            return (
                (w * 0.5 + p_lie * n_word) / (w + n_word),
                (w * 0.5 + p_true * n_word) / (w + n_word),
            )

        def score(words):
            lie = np.log(prior_lie)
            true = np.log(prior_true)
            for word in words:
                l, t = p_weighted(word)
                lie += np.log(l)
                true += np.log(t)
            return (lie, true, true > lie)

        return rdd.map(lambda row: score(row.filtered))


model = NaiveBayes()
model.fit(training.select("sender_labels", "filtered"))
print(model.predict(test).collect())

##compute accuracy and F1 of the model
