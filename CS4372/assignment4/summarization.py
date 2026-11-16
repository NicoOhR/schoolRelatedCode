import pathlib
import re
import string
import subprocess
import urllib.request
from collections import defaultdict
from operator import add

from datasets import load_dataset
from transformers import pipeline

##Summarization
frank_url = "https://www.gutenberg.org/cache/epub/84/pg84.txt"
frank_path = pathlib.Path("./data/pg84")
frank_path.parent.mkdir(parents=True, exist_ok=True)
urllib.request.urlretrieve(frank_url, frank_path)

with open(frank_path, "r") as file:
    frank_strings = file.readlines()[70:7390]


frank_chapters = []
curr_chapter = []
for line in frank_strings:
    if not re.match(r"Chapter \d+$", line.strip()):
        curr_chapter.append(line)
    else:
        frank_chapters.append(curr_chapter)
        curr_chapter = []

frank_chapters.append(curr_chapter)

frank_chapters.pop(0)

frank_chapters = ["".join(ch).replace("\n", " ").strip() for ch in frank_chapters]

# Chapter 24 ends with Walton's closing letters, which BookSum summarizes as a
# separate "final letters" section, so split them off as a 25th entry
ch24, final_letters = frank_chapters[-1].split("Walton, _in continuation._")
frank_chapters[-1] = ch24.strip()
frank_chapters.append(final_letters.strip())

booksum = load_dataset("kmfoda/booksum", split="train+validation+test").to_pandas()
booksum["title"] = booksum["book_id"].str.split(".").str[0]

frank_refs = booksum[
    (booksum["title"] == "Frankenstein") & ~booksum["is_aggregate"]
].copy()
frank_refs["chapter_num"] = frank_refs["book_id"].str.extract(r"chapter (\d+)$")[0]
# "final letters" is the only non-aggregate row without a chapter number
frank_refs["chapter_num"] = frank_refs["chapter_num"].fillna(25)
# chapter number -> list of reference summaries; frank_refs[n] goes with frank_chapters[n - 1]
frank_refs = frank_refs.groupby(frank_refs["chapter_num"].astype(int))[
    "summary_text"
].apply(list)

summarizer = pipeline("text-generation", model="facebook/bart-large-cnn")
## Translation
fr_by_heading = {}
for ebook_id in [62404, 62405, 62406]:
    fr_path = pathlib.Path(f"./data/pg{ebook_id}")
    urllib.request.urlretrieve(
        f"https://www.gutenberg.org/cache/epub/{ebook_id}/pg{ebook_id}.txt", fr_path
    )
    with open(fr_path, "r") as file:
        fr_strings = file.readlines()

    start = next(i for i, l in enumerate(fr_strings) if "START OF THE PROJECT" in l)
    end = next(i for i, l in enumerate(fr_strings) if l.startswith("FIN DU TOME"))

    heading = None
    for line in fr_strings[start:end]:
        if re.match(r"CHAPITRE [IVXL]+(er)?$", line.strip()):
            heading = line.strip()
            fr_by_heading[heading] = []
        elif heading is not None:
            fr_by_heading[heading].append(line)

fr_chapters = ["".join(ch).replace("\n", " ").strip() for ch in fr_by_heading.values()]

# The French follows the 1818 edition (23 chapters). The 1831 English split the
# 1818 chapter I into chapters 1 and 2 and rewrote much of it, so French chapter
# I has no clean match. From then on, French chapter N lines up with English N + 1.
# French XXIII includes Walton's final letters, so rejoin them to English chapter 24.
en_for_translation = frank_chapters[2:23] + [" ".join(frank_chapters[23:])]
translation_pairs = list(zip(en_for_translation, fr_chapters[1:]))
