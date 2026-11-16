import pathlib
import re
import urllib.request

import evaluate
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

N_CHAPTERS = 5  # the whole book is ~450 chunks, so only translate the first few chapters

## English source (same parsing as assignment4.py)
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

## Spanish reference translation (Gregorio Lafuerza)
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

## Translation
mt_name = "pfnet/plamo-3-translate-0.8b"
mt_tokenizer = AutoTokenizer.from_pretrained(mt_name, trust_remote_code=True)
mt_tokenizer.padding_side = "left"  # needed for batched generation, per the model card
# The T4 has no native bfloat16, and float16 can overflow in models trained in
# bfloat16. At 0.8B parameters, float32 fits easily (~3.2 GB).
mt_model = AutoModelForCausalLM.from_pretrained(
    mt_name, trust_remote_code=True, dtype="float32", device_map="auto"
)


def split_chunks(chapter, max_chunk_words=300):
    sentences = re.split(r"(?<=[.!?”])\s+", chapter)
    chunks, curr = [], []
    for s in sentences:
        if curr and len(" ".join(curr + [s]).split()) > max_chunk_words:
            chunks.append(" ".join(curr))
            curr = []
        curr.append(s)
    chunks.append(" ".join(curr))
    return chunks


# Translate all of a chapter's chunks in one batch (the model card's
# "multiple texts at once" example), then join them back together
def translate(chapter):
    chunks = split_chunks(chapter)
    inputs = mt_tokenizer.apply_chat_template(
        [[{"role": "user", "content": c}] for c in chunks],
        source_lang="en",
        target_lang="es",
        add_generation_prompt=True,
        padding=True,
        return_dict=True,
        return_tensors="pt",
    ).to(mt_model.device)
    with torch.inference_mode():
        outputs = mt_model.generate(**inputs, do_sample=False, max_new_tokens=1024)
    # every row is padded to the same prompt length, so one slice drops all the prompts
    translations = mt_tokenizer.batch_decode(
        outputs[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )
    return " ".join(t.strip() for t in translations)


ttc_translations = []
for i, (en, _) in enumerate(translation_pairs[:N_CHAPTERS]):
    ttc_translations.append(translate(en))
    print(f"translated chapter {i + 1}/{N_CHAPTERS}", flush=True)

## Evaluation
# predictions is a list of strings and references a list of reference lists, one
# per prediction. chrF is reported alongside BLEU because it matches at the
# character level, which suits a loose literary reference translation better.
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

translation_results = pd.DataFrame(
    [
        {
            "chapter": f"1.{i + 1}",
            "en_words": len(en.split()),
            "es_ref_words": len(ref.split()),
            "mt_words": len(mt.split()),
            "BLEU": bleu.compute(predictions=[mt], references=[[ref]])["score"],
            "chrF": chrf.compute(predictions=[mt], references=[[ref]])["score"],
        }
        for i, ((en, ref), mt) in enumerate(zip(translation_pairs, ttc_translations))
    ]
).set_index("chapter")

print(translation_results.round(2).to_string())
print(translation_results[["BLEU", "chrF"]].mean().round(2).to_string())

translation_results.to_csv("translation_results.csv")
# underscores in column names break LaTeX, so rename for the table only
translation_results.rename(
    columns={"en_words": "EN words", "es_ref_words": "Ref. words", "mt_words": "MT words"}
).rename_axis("Chapter").round(2).to_latex(
    "translation_results.tex",
    float_format="%.2f",
    caption="PLaMo 3 Translate 0.8B (English to Spanish) against Lafuerza's translation",
    label="tab:translation",
)
