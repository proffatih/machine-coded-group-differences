#!/usr/bin/env python3
"""Pilot: lettered options versus topic names as the response format.

Before the main run, the 21 topics were presented as lettered options (A..U) and the
logits were read over the letters. Smaller models answered from option position rather
than content. This script reproduces that comparison on one model so the figure quoted
in the manuscript is backed by a released artefact rather than a remembered number.

Design: Qwen2.5-1.5B-Instruct in 4-bit NF4; the first 100 items of four corpora;
zero-shot; (a) lettered options scored over the 21 letter tokens, (b) topic names scored
over their first tokens (the scoring used in the main study).

Usage: python code/09_pilot_response_format.py
Output: results/pilot/response_format_pilot.csv
"""
import csv, os, sys, importlib.util
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES, CAP_MAJOR_TOPICS
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "pilot"); os.makedirs(OUT, exist_ok=True)
spec = importlib.util.spec_from_file_location("ann", os.path.join(ROOT, "code", "02_annotate.py"))
ann = importlib.util.module_from_spec(spec); spec.loader.exec_module(ann)

REPO = "Qwen/Qwen2.5-1.5B-Instruct"; N = 100
CORP = ["us_nyt", "us_bills", "uk_pmqs", "us_scotus"]
LETTERS = [chr(ord("A") + i) for i in range(len(VALID_CODES))]
C = np.array(VALID_CODES)


def letter_prompt(tok, text, venue):
    opts = "\n".join(f"{l}. {CAP_MAJOR_TOPICS[c]}" for l, c in zip(LETTERS, VALID_CODES))
    msgs = [{"role": "system", "content": ann.SYSTEM},
            {"role": "user", "content": f"Assign the following text to exactly one CAP major topic.\n\n"
             f"Source: {venue}\n\nTopics:\n{opts}\n\nText: \"{text}\"\n\n"
             f"Answer with the single letter of the correct topic and nothing else."}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def letter_ids(tok):
    out = []
    for l in LETTERS:
        ids = tok.encode(l, add_special_tokens=False)
        if len(ids) != 1:
            ids = tok.encode(" " + l, add_special_tokens=False)
        out.append(ids[0])
    return out


@torch.no_grad()
def score(model, tok, prompts, ids, bs=4):
    idx = torch.tensor(ids, device="cuda"); res = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        lg = model(**enc, logits_to_keep=1).logits[:, -1, :]
        res.append(lg.index_select(1, idx).float().argmax(1).cpu().numpy())
    return np.concatenate(res)


def position_profile(choice_idx):
    """Share of answers on the first three options, against 3/21 if answers ignored order."""
    return float(np.mean(choice_idx < 3))


def main():
    tok = AutoTokenizer.from_pretrained(REPO); tok.padding_side = "left"
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(REPO, device_map="cuda", attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)).eval()
    first_ids, _ = ann.label_prefixes(tok)
    rows = []
    for ck in CORP:
        data = list(csv.DictReader(open(os.path.join(ROOT, "data", f"{ck}.csv"), encoding="utf-8")))[:N]
        gold = np.array([int(r["label"]) for r in data]); venue = CORPORA[ck][5]
        pl = score(model, tok, [letter_prompt(tok, r["text"], venue) for r in data], letter_ids(tok))
        pn = score(model, tok, [ann.build_prompt(tok, r["text"], venue, "zs", []) for r in data], first_ids)
        a_l, a_n = float((C[pl] == gold).mean()), float((C[pn] == gold).mean())
        true_first3 = float(np.mean(np.searchsorted(C, gold) < 3))
        rows.append({"corpus": ck, "n": N, "letter_accuracy": a_l, "name_accuracy": a_n,
                     "ratio": a_n / a_l if a_l else float("nan"),
                     # how often each format picks one of the first three listed topics,
                     # against how often one of those three is actually the right answer
                     "letter_share_first3": position_profile(pl),
                     "name_share_first3": position_profile(pn),
                     "gold_share_first3": true_first3})
        print(f"  {ck:10s} harf={a_l:.3f} ad={a_n:.3f} oran={a_n/a_l:.2f}  ilk-3 secim: harf={position_profile(pl):.2f} ad={position_profile(pn):.2f} dogru={true_first3:.2f}", flush=True)
    with open(os.path.join(OUT, "response_format_pilot.csv"), "w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    ml = np.mean([r["letter_accuracy"] for r in rows]); mn = np.mean([r["name_accuracy"] for r in rows])
    print(f"  ORTALAMA harf={ml:.3f} ad={mn:.3f} oran={mn/ml:.2f}  yazildi -> results/pilot/")


if __name__ == "__main__":
    main()
