import pathlib
import re
import urllib.request

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

## English chapters (same parsing as assignment4.py)
ttc_url = "https://www.gutenberg.org/cache/epub/98/pg98.txt"
ttc_path = pathlib.Path("./data/pg98")
ttc_path.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(ttc_url, ttc_path)

# The book proper begins at line 100 and the Gutenberg licence starts at line 15933
with open(ttc_path, "r") as file:
    ttc_strings = file.readlines()[99:15932]


ttc_chapters = []
curr_chapter = []
for line in ttc_strings:
    if line.startswith("Book the"):
        continue
    if not re.match(r"CHAPTER [IVXL]+\.$", line.strip()):
        curr_chapter.append(line)
    else:
        ttc_chapters.append(curr_chapter)
        curr_chapter = []

ttc_chapters.append(curr_chapter)

ttc_chapters.pop(0)

ttc_chapters = ["".join(ch).replace("\n", " ").strip() for ch in ttc_chapters]

# book.chapter labels: the three books have 6, 24 and 15 chapters
chapter_labels = [
    f"{book}.{ch}" for book, n in [(1, 6), (2, 24), (3, 15)] for ch in range(1, n + 1)
]

## Passages
# Whole chapters (1.3k-7.9k tokens) make poor documents: a short question matches
# every chapter weakly, and the reader has to find a span in thousands of words.
# Split each chapter into overlapping passages of 200 words instead.
PASSAGE_WORDS = 200
OVERLAP = 50

passages, passage_chapter = [], []
for label, chapter in zip(chapter_labels, ttc_chapters):
    words = chapter.split()
    for start in range(0, max(len(words) - OVERLAP, 1), PASSAGE_WORDS - OVERLAP):
        passages.append(" ".join(words[start : start + PASSAGE_WORDS]))
        passage_chapter.append(label)

print(f"{len(passages)} passages from {len(ttc_chapters)} chapters")

## TF-IDF search, one document per passage
vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
doc_matrix = vectorizer.fit_transform(passages)


def search(query, k=3):
    sims = cosine_similarity(vectorizer.transform([query]), doc_matrix)[0]
    top = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]


## Extractive question answering
# deepset/roberta-base-squad2 is RoBERTa fine-tuned on SQuAD 2.0 (questions about
# Wikipedia paragraphs), so it expects exactly this: a question and a short passage.
# transformers 5 dropped the plain "question-answering" pipeline, so the answer span
# is read off the model's start/end logits directly.
qa_name = "deepset/roberta-base-squad2"
device = "cuda" if torch.cuda.is_available() else "cpu"
qa_tokenizer = AutoTokenizer.from_pretrained(qa_name)
qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_name).to(device).eval()


def read(question, passage, max_answer_tokens=30):
    enc = qa_tokenizer(
        question,
        passage,
        truncation="only_second",
        max_length=512,
        return_offsets_mapping=True,
        return_tensors="pt",
    )
    offsets = enc.pop("offset_mapping")[0]
    context = torch.tensor([s == 1 for s in enc.sequence_ids(0)])  # passage tokens only
    with torch.inference_mode():
        out = qa_model(**enc.to(device))
    start_logp = out.start_logits[0].cpu().log_softmax(-1)
    end_logp = out.end_logits[0].cpu().log_softmax(-1)

    # best span with start <= end, both inside the passage, at most 30 tokens long
    scores = start_logp[:, None] + end_logp[None, :]
    n = scores.shape[0]
    valid = torch.ones(n, n, dtype=torch.bool).triu() & ~torch.ones(n, n, dtype=torch.bool).triu(max_answer_tokens)
    valid &= context[:, None] & context[None, :]
    scores[~valid] = float("-inf")
    best = int(scores.argmax())
    s, e = divmod(best, n)
    return {
        "answer": passage[int(offsets[s][0]) : int(offsets[e][1])],
        "score": float(scores[s, e].exp()),
    }


# Ask the question of each retrieved passage and keep the most confident answer
def answer(question, k=3):
    hits = search(question, k)
    best = None
    for idx, sim in hits:
        pred = read(question, passages[idx])
        if best is None or pred["score"] > best["score"]:
            best = {**pred, "chapter": passage_chapter[idx]}
    return hits, best


# Questions, the expected answer, and a key phrase: retrieval counts as a hit if a
# retrieved passage contains it, and the answer counts as correct if it does
questions = [
    ("What is the name of the bank Mr. Lorry works for?", "Tellson's", "Tellson"),
    ("What number was Doctor Manette's cell in the Bastille?", "One Hundred and Five, North Tower", "One Hundred and Five"),
    ("Who was tried for treason at the Old Bailey?", "Charles Darnay", "Darnay"),
    ("What does Madame Defarge do while she sits in the wine-shop?", "knits", "knit"),
    ("Who takes Darnay's place at the guillotine?", "Sydney Carton", "Carton"),
]

rows = []
for question, expected, key in questions:
    hits, best = answer(question)
    rows.append(
        {
            "Question": question,
            "Expected": expected,
            "Top 3 passages: ch. (cos sim)": ", ".join(
                f"{passage_chapter[i]} ({s:.2f})" for i, s in hits
            ),
            "Answer in top 3": any(key.lower() in passages[i].lower() for i, _ in hits),
            "Answer": best["answer"],
            "Answer ch.": best["chapter"],
            "Score": best["score"],
            "Correct": key.lower() in best["answer"].lower(),
        }
    )
    print(question, "->", best["answer"], f"({best['chapter']}, {best['score']:.2f})", flush=True)

qa_results = pd.DataFrame(rows)
print(qa_results.to_string())
print("retrieval hits:", qa_results["Answer in top 3"].sum(), "/ 5,",
      "correct answers:", qa_results["Correct"].sum(), "/ 5")

# the top passage for each question, to show what the reader saw
for question, _, _ in questions:
    idx, sim = search(question, 1)[0]
    print(f"\n{question}\n[{passage_chapter[idx]}, {sim:.2f}] {passages[idx]}")

qa_results.to_csv("qa_results.csv", index=False)
qa_results.to_latex(
    "qa_results.tex",
    index=False,
    float_format="%.2f",
    column_format=r">{\raggedright\arraybackslash}p{3.3cm}>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}p{2.6cm}>{\raggedright\arraybackslash}p{1.1cm}>{\raggedright\arraybackslash}p{2.4cm}>{\raggedright\arraybackslash}p{0.9cm}rl",
    caption="TF-IDF retrieval of the top 3 200-word passages, then answers from "
    "deepset/roberta-base-squad2",
    label="tab:qa",
)
