import pathlib
import subprocess

import numpy as np
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.sql import Row
from pyspark.sql.session import SparkSession
from pyspark.sql.types import *

friend_path = pathlib.Path("./soc-LiveJournal1Adj.txt")

if not friend_path.is_file():
    subprocess.run(
        ["wget", "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"],
        check=True,
    )

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.12.3")
    .config("spark.driver.memory", "20g")
    .getOrCreate()
)
sc = spark.sparkContext

# (user, [friends])
text_file = (
    sc.textFile(str(friend_path))
    .map(lambda x: x.split("\t"))
    .map(lambda x: (x[0], x[1]))
    .mapValues(lambda x: x.split(","))
)


def getFriendsRecFstOrder(user):
    """
    Two users should be reccomended to each other if they have many friends in common
    and are not friends with each other. Take a userID as input, call him user a for
    each of their friends, get their friends, filter users that user a is already friends
    with the most frequently appearing user should then be reccomended to user a. This is the
    first order reccomendation, higher order reccomendations, where the list from friends
    of the friends of a, can be pulled in.
    """
    friends_list = [x for xs in text_file.lookup(user) for x in xs]
    friends_list.append(user)
    neighbors = (
        text_file.filter(lambda x: x[0] in friends_list)
        .flatMap(lambda x: x[1])
        .map(lambda x: (x, 1))
        .reduceByKey(lambda x, y: x + y)
        .filter(lambda x: x[0] not in friends_list)
    )
    return neighbors.takeOrdered(10, lambda x: -x[1])


# (user, user_friend, 1)
sparse = text_file.flatMap(lambda x: [(x[0], x1) for x1 in x[1]]).map(
    lambda x: (x[0], x[1], 1)
)
sparse_df = spark.createDataFrame(sparse, ["user", "user_friend", "connected"])
training, test = sparse_df.randomSplit([0.8, 0.2])

indexer = StringIndexer(
    inputCols=["user", "user_friend"], outputCols=["userIdx", "user_friendIdx"]
).setHandleInvalid("keep")
als = ALS(
    rank=10,
    seed=0,
    userCol="userIdx",
    itemCol="user_friendIdx",
    ratingCol="connected",
    coldStartStrategy="drop",
    implicitPrefs=True,
)

pipeline = Pipeline(stages=[indexer, als])
evaluator = RegressionEvaluator(
    metricName="rmse", labelCol="connected", predictionCol="prediction"
)

grid = (
    ParamGridBuilder()
    .addGrid(als.rank, [5, 10, 20])
    .addGrid(als.regParam, [0.01, 0.1, 0.5])
    .addGrid(als.maxIter, [10, 20])
    .build()
)

crossval = CrossValidator(
    estimator=pipeline, estimatorParamMaps=grid, numFolds=2, evaluator=evaluator
)

cvModel = crossval.fit(training)
best_model = cvModel.bestModel
best_model.stages[-1].rank
print(best_model.rank)
predictions = best_model.transform(test)
test.printSchema()
rmse = evaluator.evaluate(predictions)
print(rmse)

recs = best_model.stages[-1].recommendForAllUsers(10)

random = [
    x[0]
    for x in training.sample(
        withReplacement=False, fraction=(30 / text_file.count()), seed=2
    ).take(10)
]

als_rec = recs.filter(recs.userIdx.isin(random))

schema = StructType(
    [
        StructField("userIdx", IntegerType(), True),
        StructField(
            "mapReduceRecs",
            ArrayType(
                StructType(
                    [
                        StructField("friend", StringType(), True),
                        StructField("reccScore", IntegerType(), True),
                    ]
                )
            ),
            True,
        ),
    ]
)
map_reduce_rec = sc.parallelize(
    list(map(lambda x: (int(x), getFriendsRecFstOrder(x)), random))
).toDF(schema=schema)

combined = als_rec.join(map_reduce_rec, "userIdx")
combined.head()
