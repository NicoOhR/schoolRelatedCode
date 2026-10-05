import spacy
from operator import add
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
    subprocess.run(
        ["wget", "-w", "2", "-m", "-H", "https://www.gutenberg.org/cache/epub/25717/pg25717.txt"],
        check=True,
    )

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

movie_path = pathlib.Path("./MovieSummaries/plot_summaries.txt")

if not movie_path.is_file(): 
    subprocess.run(
        ["wget", "-nc", "http://www.cs.cmu.edu/~ark/personas/data/MovieSummaries.tar.gz"],
        check=True,
    )
    subprocess.run(["tar", "-xzf", "MovieSummaries.tar.gz"], check=True)

def reduce_inner(words):
    acc = defaultdict(int)
    for k in words: 
        acc[k] += 1
    return list(acc.items())

## (word, [(frequency, id)])
docs = (sc.textFile(str(movie_path))
       .map(lambda x: "".join([char for char in x if char not in string.punctuation]))
       .map(lambda x: re.split(r'[\s\t,]', x))
       .map(lambda x: (x[0], x[1:]))
       .mapValues(lambda x: [w for w in x if len(w) > 0])
       .mapValues(lambda x: [w.lower() for w in x])
       .mapValues(reduce_inner)
       )

n_docs = docs.count()

meta = sc.textFile("./MovieSummaries/movie.metadata.tsv").map(lambda x: x.split("\t")).map(lambda x: (x[0], x[2]))

rdd = (docs
       .flatMap(lambda x: [(w[0], (x[0],w[1])) for w in x[1]])
       .groupByKey()
       .mapValues(list)
       )

def single_word_search(word):
    res = rdd.lookup(word.lower())
    if res:
        word_rdd = sc.parallelize([(id, freq) for id, freq in res[0]])
        n_word = word_rdd.count()
        scores = word_rdd.mapValues(lambda x: x * np.log(n_docs / n_word))
        return scores.join(meta).sortBy(lambda x: x[1], ascending=False).map(lambda x: x[1][1]).take(10)
    else:
        return None
   
def inverse_document_freq(word): 
    return np.log(n_docs/len(rdd.lookup(word)))

def multiple_word_search(query):
    query = query.lower()
    query_freq = defaultdict(float)
    query_inverse = defaultdict(float)

    for token in query.split():
        query_freq[token] += 1
        query_inverse[token] = inverse_document_freq(token)

    query_freq = {key: value/len(query_freq.keys()) for key, value in query_freq.items()} 
    query_vec = {k: query_freq[k] * query_inverse[k] for k in query_freq.keys()}

    query_norm = np.sqrt(sum(v * v for v in query_vec.values()))

    q = sc.broadcast(query_vec)
   
    tfidf = rdd.flatMap(lambda x: [(doc, (x[0], freq*np.log(n_docs/len(x[1])))) for doc, freq in x[1]])
    doc_norms = (tfidf.mapValues(lambda wt: wt[1] ** 2).reduceByKey(add).mapValues(np.sqrt))

    dots = tfidf.filter(lambda d: d[1][0] in q.value).mapValues(lambda wt: wt[1] * q.value[wt[0]]).reduceByKey(add)

    scores = dots.join(doc_norms).mapValues(lambda dn: dn[0] / (dn[1] * query_norm))

    return scores.join(meta).sortBy(lambda x: x[1], ascending=False).map(lambda x: x[1][1]).take(10)

search_path = pathlib.Path("./search_terms.txt")

## one query per line: a single word runs single_word_search, anything longer runs multiple_word_search
queries = [line.strip() for line in search_path.read_text().splitlines() if line.strip()]

for query in queries:
    if len(query.split()) == 1:
        results = single_word_search(query)
        kind = "single"
    else:
        results = multiple_word_search(query)
        kind = "multiple"
    print(f"[{kind}] {query}")
    for rank, title in enumerate(results or [], start=1):
        print(f"  {rank:2}. {title}")
    if not results:
        print("  no results")
    print()
