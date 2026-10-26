import pathlib
import subprocess

import pyspark
from pyspark.sql.session import SparkSession

friend_path = pathlib.Path("./soc-LiveJournal1Adj.txt")

if not friend_path.is_file():
    subprocess.run(
        ["wget", "https://an-ml.s3.us-west-1.amazonaws.com/soc-LiveJournal1Adj.txt"],
        check=True,
    )

spark = (
    SparkSession.builder.master("local[*]")
    .config("spark.jars.packages", "io.graphframes:graphframes-spark4_2.13:0.12.3")
    .getOrCreate()
)
sc = spark.sparkContext

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
    print(neighbors.takeOrdered(10, lambda x: -x[1]))


getFriendsRecFstOrder("0")
