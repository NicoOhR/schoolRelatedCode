import spacy
from collections import defaultdict
import string
from pyspark.sql.session import SparkSession
import pyspark
import subprocess
import pathlib
import pyspark
import numpy as np
from spacy.pipeline.ner  import DEFAULT_NER_MODEL
import re

### Q1 
gibon_path = pathlib.Path("./www.gutenberg.org/cache/epub/25717/pg25717.txt")

if not gibon_path.is_file(): 
    subprocess.call("wget -w 2 -m -H \"https://www.gutenberg.org/cache/epub/25717/pg25717.txt\"")

sc = pyspark.SparkContext().getOrCreate()
spark = SparkSession.builder.getOrCreate()

config = {
    "moves": None, 
    "update_with_oracle_cut_size": 100, 
    "model": DEFAULT_NER_MODEL, 
    "incorrect_spans_key": "incorrect_spans",
}


def get_named_entities(lines):
    nlp = spacy.load("en_core_web_sm", exclude=["tagger", "parser", "attribute_ruler", "lemmatizer"])
    for doc in nlp.pipe(lines):
        yield [(ent.text, ent.label_) for ent in doc.ents]

ROMAN_RE = re.compile(
    r"^(?=[mdclxvi])m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$",
    re.IGNORECASE,
)

def strip_citations(name: str) -> str:
    kept = []
    for word in re.split(r"[\s—]+", name): 
        w = word.strip(".,;:()")
        if ROMAN_RE.match(w):
            if w.isupper() and kept and kept[-1].lower() not in ("chapter", "part") and not word.endswith("."):
                kept.append(w)
            continue
        if len(w) <= 1 or w.isdigit():
            continue 
        kept.append(w)
    return " ".join(kept)


rdd = sc.textFile(str(gibon_path))
entities = rdd.mapPartitions(get_named_entities).flatMap(lambda x:x)
people = entities.filter(lambda x: x[1] == 'PERSON').map(lambda x: strip_citations(x[0]).lower()).filter(lambda x: x).map(lambda x: (x, 1))
count = people.reduceByKey(lambda x, y: x + y).sortBy(lambda x: x[1], ascending=False).collect()
count

###Q2

sc = pyspark.SparkContext().getOrCreate()
spark = SparkSession.builder.getOrCreate()

movie_path = pathlib.Path("./MovieSummaries/plot_summaries.txt")

if not movie_path.is_file(): 
    subprocess.call("wget \"http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz\"")

def reduce_inner(words):
    acc = defaultdict(int)
    for k in words: 
        acc[k] += 1
    return list(acc.items())

## (word, [(frequency, id)])
docs = (sc.textFile(str(movie_path))
       .map(lambda x: "".join([char for char in x if char not in string.punctuation]))
       .map(lambda x: re.split(r'[\s\t,]', x))
       .map(lambda x: (x[0], x[1:-1]))
       .mapValues(lambda x: [w for w in x if len(w) > 0])
       .mapValues(lambda x: [w.lower() for w in x])
       .mapValues(reduce_inner)
       )

n_docs = docs.count()

rdd = (docs
       .flatMap(lambda x: [(w[0], (x[0],w[1])) for w in x[1]])
       .groupByKey()
       .mapValues(list)
       )

def single_word_search(word):
    res = rdd.lookup(word)
    if res:
        word_rdd = sc.parallelize([(id, freq) for id, freq in res[0]])
        n_word = word_rdd.count()
        return word_rdd.mapValues(lambda x: x * np.log(n_docs / n_word)).takeOrdered(10, lambda x: -x[1])
        # return word_rdd.reduceByKey()
    else:
        return None
    

# def multiple_word_search(sentence):

