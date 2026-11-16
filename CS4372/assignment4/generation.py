import math
import pathlib
import re
import urllib.request

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

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

# Keep the line breaks this time: the chapter title is the first non-empty line
chapter_lines = ["".join(ch).strip().split("\n", 1) for ch in ttc_chapters]
chapter_titles = [lines[0].strip() for lines in chapter_lines]
chapter_bodies = [" ".join(lines[1].split()) for lines in chapter_lines]

chapter_labels = [
    f"{book}.{ch}" for book, n in [(1, 6), (2, 24), (3, 15)] for ch in range(1, n + 1)
]

## Models
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPT-2 generates; GPT-2 Large scores. Scoring with a different, bigger model avoids
# the generator grading its own output, which would make its perplexity look low.
# Both use the same tokenizer, so token counts line up.
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
generator = AutoModelForCausalLM.from_pretrained("openai-community/gpt2").to(device).eval()
scorer = AutoModelForCausalLM.from_pretrained("openai-community/gpt2-large").to(device).eval()

SEED_WORDS = 20  # "a small seed": the first 20 words of a chapter
NEW_TOKENS = 60

# Chapters whose openings aren't famous, so GPT-2 can't just recite the book
# (it has memorized "It was the best of times...")
seed_chapters = ["1.2", "2.1", "2.7", "3.1", "3.5"]


def generate(seed):
    inputs = tokenizer(seed, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = generator.generate(
            **inputs,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            no_repeat_ngram_size=3,
            max_new_tokens=NEW_TOKENS,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)


# Perplexity of a continuation given the seed: the seed is context only, so its
# tokens are masked out of the loss with -100
def perplexity(seed, continuation):
    seed_ids = tokenizer(seed, return_tensors="pt")["input_ids"]
    cont_ids = tokenizer(continuation, return_tensors="pt")["input_ids"]
    ids = torch.cat([seed_ids, cont_ids], dim=1).to(device)
    labels = ids.clone()
    labels[:, : seed_ids.shape[1]] = -100
    with torch.inference_mode():
        loss = scorer(input_ids=ids, labels=labels).loss
    return math.exp(loss.item())


torch.manual_seed(0)
rows = []
for label in seed_chapters:
    body = chapter_bodies[chapter_labels.index(label)]
    words = body.split()
    seed = " ".join(words[:SEED_WORDS])

    generated = generate(seed)
    n_gen = len(tokenizer(generated)["input_ids"])
    # the real next stretch of the book, cut to the same number of tokens
    book_ids = tokenizer(" " + " ".join(words[SEED_WORDS:]))["input_ids"][:n_gen]
    book_continuation = tokenizer.decode(book_ids)

    rows.append(
        {
            "Ch.": label,
            "Seed": seed,
            "Generated continuation": " ".join(generated.split()),
            "PPL (generated)": perplexity(seed, generated),
            "PPL (book)": perplexity(seed, book_continuation),
            "Book continuation": " ".join(book_continuation.split()),
        }
    )
    print(label, "|", seed, "||", generated, flush=True)

generation_results = pd.DataFrame(rows)
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
