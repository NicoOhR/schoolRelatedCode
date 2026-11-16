import math
import pathlib
import re
import string
import subprocess
import urllib.request
from collections import defaultdict
from operator import add

import matplotlib.pyplot as plt
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

##Summarization
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

model_name = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

ttc_token_lengths = [
    len(tokenizer(ch, add_special_tokens=False)["input_ids"]) for ch in ttc_chapters
]


# split each chapter into even chunks, summarize each one, and join the results
def summarize(text, max_chunk_tokens=1000):
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    n_chunks = math.ceil(len(ids) / max_chunk_tokens)
    size = math.ceil(len(ids) / n_chunks)
    chunks = [tokenizer.decode(ids[i : i + size]) for i in range(0, len(ids), size)]

    batch = tokenizer(
        chunks, return_tensors="pt", padding=True, truncation=True, max_length=1024
    ).to(device)
    with torch.inference_mode():
        out = model.generate(
            **batch, num_beams=4, min_new_tokens=30, max_new_tokens=128
        )
    return " ".join(tokenizer.batch_decode(out, skip_special_tokens=True))


ttc_summaries = [summarize(ch) for ch in ttc_chapters]

# reference chapter summaries to test against with ROUGE and BERTScore
booksum = load_dataset("kmfoda/booksum", split="train+validation+test").to_pandas()
booksum["title"] = booksum["book_id"].str.split(".").str[0]

ttc_refs = booksum[
    (booksum["title"] == "A Tale of Two Cities") & ~booksum["is_aggregate"]
].copy()
ttc_refs[["book", "chapter_num"]] = (
    ttc_refs["book_id"].str.extract(r"book (\d+)\.chapter (\d+)$").astype(int)
)
# (book, chapter) -> list of reference summaries, in reading order,
# ttc_refs.iloc[i] <-> ttc_chapters[i]
ttc_refs = ttc_refs.groupby(["book", "chapter_num"])["summary_text"].apply(list)

ttc_refs_list = ttc_refs.tolist()
assert len(ttc_refs_list) == len(ttc_chapters)

# Evaluation
from bert_score import BERTScorer
from rouge import Rouge

rouge = Rouge()


# Rouge.get_scores(hyps, refs) pairs hyps[i] with refs[i], so repeat the summary once
# per reference.
def get_rouge_score(summary, refs):
    scores = rouge.get_scores([summary] * len(refs), refs)
    return {
        m: max(s[m]["f"] for s in scores) for m in ["rouge-1", "rouge-2", "rouge-l"]
    }


rouge_scores_out = [
    get_rouge_score(ttc_summaries[i], ttc_refs_list[i])
    for i in range(len(ttc_chapters))
]

# Passing a list of reference lists scores each summary against all of its references
# and keeps the best. Rescaling spreads the scores out; raw BERTScore F1 sits around 0.85.
scorer = BERTScorer(lang="en", rescale_with_baseline=True)
bert_p, bert_r, bert_f1 = scorer.score(ttc_summaries, ttc_refs_list)

results = pd.DataFrame(rouge_scores_out)
results["bertscore"] = bert_f1.tolist()
results["tokens"] = ttc_token_lengths
results.index = [f"{book}.{ch}" for book, ch in ttc_refs.index]
print(results.round(3))
print(results.mean().round(3))

fig, (ax_scores, ax_tokens) = plt.subplots(
    2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
)
x = range(len(results))
for col, label, color in [
    ("rouge-1", "ROUGE-1", "#2a78d6"),
    ("rouge-2", "ROUGE-2", "#eb6834"),
    ("rouge-l", "ROUGE-L", "#1baf7a"),
    ("bertscore", "BERTScore (rescaled)", "#eda100"),
]:
    ax_scores.plot(
        x, results[col], label=label, color=color, linewidth=2, marker="o", markersize=4
    )
ax_scores.set_ylabel("F1 vs. best reference")
ax_scores.set_title("A Tale of Two Cities: BART summary scores by chapter")
ax_scores.legend(loc="upper left", ncols=4, frameon=False)

ax_tokens.plot(
    x, results["tokens"], color="#52514e", linewidth=2, marker="o", markersize=4
)
ax_tokens.set_ylabel("Chapter length (tokens)")
ax_tokens.set_xlabel("Chapter (book.chapter)")
ax_tokens.set_xticks(list(x), results.index, rotation=90)

for ax in (ax_scores, ax_tokens):
    ax.grid(axis="y", color="#e5e4e0", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    # mark where books 2 and 3 start
    for start in (6, 30):
        ax.axvline(start - 0.5, color="#b5b4ae", linewidth=1, linestyle="--")

fig.tight_layout()
fig.savefig("summary_scores.png", dpi=150)
plt.show()

# Save the scores and the summaries themselves for the report
results.to_csv("summary_results.csv", index_label="chapter")
pd.DataFrame(
    {"chapter": results.index, "summary": ttc_summaries, "references": ttc_refs_list}
).to_json("summaries.json", orient="records", indent=1)
results.rename(
    columns={
        "rouge-1": "ROUGE-1",
        "rouge-2": "ROUGE-2",
        "rouge-l": "ROUGE-L",
        "bertscore": "BERTScore",
        "tokens": "Tokens",
    }
).rename_axis("Ch.").to_latex(
    "summary_results.tex",
    float_format="%.3f",
    longtable=True,
    caption="BART-large-CNN chapter summaries scored against the best-matching "
    "BookSum reference (F1). BERTScore is rescaled with its baseline.",
    label="tab:summary",
)

## Translation
es_url = "https://www.gutenberg.org/cache/epub/61887/pg61887.txt"
es_path = pathlib.Path("./data/pg61887")
urllib.request.urlretrieve(es_url, es_path)

# The book proper begins at line 117; line 17310 is "FIN", followed by the table of contents
with open(es_path, "r") as file:
    es_strings = file.readlines()[116:17309]


es_chapters = []
curr_chapter = []
for line in es_strings:
    if line.startswith("LIBRO"):
        continue
    if not re.match(r"[IVXL]+\.?$", line.strip()):
        curr_chapter.append(line)
    else:
        es_chapters.append(curr_chapter)
        curr_chapter = []

es_chapters.append(curr_chapter)

es_chapters.pop(0)

es_chapters = ["".join(ch).replace("\n", " ").strip() for ch in es_chapters]

translation_pairs = list(zip(ttc_chapters, es_chapters))

from transformers import AutoModelForCausalLM, AutoTokenizer

mt_tokenizer = AutoTokenizer.from_pretrained(
    "pfnet/plamo-3-translate-0.8b", trust_remote_code=True
)
mt_model = AutoModelForCausalLM.from_pretrained(
    "pfnet/plamo-3-translate-0.8b",
    trust_remote_code=True,
    dtype="bfloat16",
    device_map="auto",
)


def translate_chunk(text):
    inputs = mt_tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        target_lang="es",
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(mt_model.device)
    with torch.inference_mode():
        output = mt_model.generate(**inputs, do_sample=False, max_new_tokens=1024)
    return mt_tokenizer.decode(
        output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def translate(chapter, max_chunk_words=300):
    sentences = re.split(r"(?<=[.!?”])\s+", chapter)
    chunks, curr = [], []
    for s in sentences:
        if curr and len(" ".join(curr + [s]).split()) > max_chunk_words:
            chunks.append(" ".join(curr))
            curr = []
        curr.append(s)
    chunks.append(" ".join(curr))
    return " ".join(translate_chunk(c) for c in chunks)


ttc_translation = [translate(en) for en, _ in translation_pairs[:5]]

# Evaluate
import evaluate

metric = evaluate.load("sacrebleu")
results = [
    metric.compute(predictions=translation, reference=ref)
    for translation, ref in zip(translation_pairs, es_chapters[5:])
]

## Question answering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForQuestionAnswering

# book.chapter labels: the three books have 6, 24 and 15 chapters
chapter_labels = [
    f"{book}.{ch}" for book, n in [(1, 6), (2, 24), (3, 15)] for ch in range(1, n + 1)
]

# passages are used as the base document size for tf-idf search
PASSAGE_WORDS = 200
OVERLAP = 50

passages, passage_chapter = [], []
for label, chapter in zip(chapter_labels, ttc_chapters):
    words = chapter.split()
    for start in range(0, max(len(words) - OVERLAP, 1), PASSAGE_WORDS - OVERLAP):
        passages.append(" ".join(words[start : start + PASSAGE_WORDS]))
        passage_chapter.append(label)

# TF-IDF search, one document per passage
vectorizer = TfidfVectorizer(stop_words="english", sublinear_tf=True)
doc_matrix = vectorizer.fit_transform(passages)


def search(query, k=3):
    sims = cosine_similarity(vectorizer.transform([query]), doc_matrix)[0]
    top = sims.argsort()[::-1][:k]
    return [(int(i), float(sims[i])) for i in top]


qa_name = "deepset/roberta-base-squad2"
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
    span_scores = start_logp[:, None] + end_logp[None, :]
    n = span_scores.shape[0]
    valid = torch.ones(n, n, dtype=torch.bool).triu() & ~torch.ones(
        n, n, dtype=torch.bool
    ).triu(max_answer_tokens)
    valid &= context[:, None] & context[None, :]
    span_scores[~valid] = float("-inf")
    s, e = divmod(int(span_scores.argmax()), n)
    return {
        "answer": passage[int(offsets[s][0]) : int(offsets[e][1])],
        "score": float(span_scores[s, e].exp()),
    }


def answer(question, k=3):
    hits = search(question, k)
    best = None
    for idx, sim in hits:
        pred = read(question, passages[idx])
        if best is None or pred["score"] > best["score"]:
            best = {**pred, "chapter": passage_chapter[idx]}
    return hits, best


questions = [
    ("What is the name of the bank Mr. Lorry works for?", "Tellson's", "Tellson"),
    (
        "What number was Doctor Manette's cell in the Bastille?",
        "One Hundred and Five, North Tower",
        "One Hundred and Five",
    ),
    ("Who was tried for treason at the Old Bailey?", "Charles Darnay", "Darnay"),
    ("What does Madame Defarge do while she sits in the wine-shop?", "knits", "knit"),
    ("Who takes Darnay's place at the guillotine?", "Sydney Carton", "Carton"),
]

qa_rows = []
for question, expected, key in questions:
    hits, best = answer(question)
    qa_rows.append(
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

qa_results = pd.DataFrame(qa_rows)
print(qa_results.to_string())

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

## Text generation
# The chapter title is the text before the first run of spaces left by blank lines
chapter_bodies = [re.split(r"\s{2,}", ch, maxsplit=1)[1] for ch in ttc_chapters]

# GPT-2 is used for generation and GPT-2 Large scores.
gen_tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
gen_model = (
    AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device).eval()
)
ppl_model = (
    AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large")
    .to(device)
    .eval()
)

SEED_WORDS = 20
NEW_TOKENS = 60

seed_chapters = ["1.2", "2.1", "2.7", "3.1", "3.5"]


def generate(seed):
    inputs = gen_tokenizer(seed, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = gen_model.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3,
            max_new_tokens=NEW_TOKENS,
            pad_token_id=gen_tokenizer.eos_token_id,
        )
    return gen_tokenizer.decode(
        out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def perplexity(seed, continuation):
    seed_ids = gen_tokenizer(seed, return_tensors="pt")["input_ids"]
    cont_ids = gen_tokenizer(continuation, return_tensors="pt")["input_ids"]
    ids = torch.cat([seed_ids, cont_ids], dim=1).to(device)
    labels = ids.clone()
    labels[:, : seed_ids.shape[1]] = -100
    with torch.inference_mode():
        loss = ppl_model(input_ids=ids, labels=labels).loss
    return math.exp(loss.item())


torch.manual_seed(0)
gen_rows = []
for label in seed_chapters:
    words = chapter_bodies[chapter_labels.index(label)].split()
    seed = " ".join(words[:SEED_WORDS])

    generated = generate(seed)
    n_gen = len(gen_tokenizer(generated)["input_ids"])
    book_ids = gen_tokenizer(" " + " ".join(words[SEED_WORDS:]))["input_ids"][:n_gen]
    book_continuation = gen_tokenizer.decode(book_ids)

    gen_rows.append(
        {
            "Ch.": label,
            "Seed": seed,
            "Generated continuation": " ".join(generated.split()),
            "PPL (generated)": perplexity(seed, generated),
            "PPL (book)": perplexity(seed, book_continuation),
            "Book continuation": " ".join(book_continuation.split()),
        }
    )

generation_results = pd.DataFrame(gen_rows)
print(generation_results[["Ch.", "PPL (generated)", "PPL (book)"]].round(2).to_string())
print(generation_results[["PPL (generated)", "PPL (book)"]].mean().round(2).to_string())

generation_results.to_csv("generation_results.csv", index=False)
generation_results.drop(columns="Book continuation").to_latex(
    "generation_results.tex",
    index=False,
    float_format="%.2f",
    column_format=r"l>{\raggedright\arraybackslash}p{3.6cm}>{\raggedright\arraybackslash}p{7.0cm}rr",
    caption="GPT-2 continuations of 20-word seeds from the book. Perplexity is "
    "scored by GPT-2 Large on the continuation given the seed, for the generated "
    "text and for the book's own next tokens.",
    label="tab:generation",
)
