#!/usr/bin/env python3
"""Annotate every CAP corpus with each open-weight LLM under each prompt condition.

The model never generates free text. The 21 CAP topics are listed by name, and the
score for each topic is read from the logits at the answer position. One forward pass
therefore yields a well-defined 21-way response distribution, and every item receives a
response -- no refusals, no unparsable output, no silent attrition that would itself
differ by subgroup and contaminate the invariance tests downstream.

Why by name rather than by letter. An earlier version presented the topics as lettered
options and read the logits over "A".."U". Small models handled that badly: they
answered from position rather than content, piling probability onto the first few
letters, and zero-shot accuracy sat near chance. Scoring the topic *name* instead
raised zero-shot accuracy on the same model and corpora by a factor of three to four
(0.10 -> 0.46 on newspaper headlines, 0.12 -> 0.32 on bill titles). The response format
is part of the instrument, and the lettered format was measuring the wrong thing.

Names are separated by their first token where that is unique, which it is for five of
the six tokenizers. Where two labels share a first token -- Granite merges "Civil" and
"Cultural" -- the pair is resolved with one extra forward pass on the second token.

Results are written per (model, condition, corpus) shard so the run is resumable.

Usage:
    python code/02_annotate.py                       # everything still missing
    python code/02_annotate.py --models qwen2.5-1.5b --conditions zs
"""
import argparse, csv, gc, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cap_codebook import CORPORA, VALID_CODES, CAP_MAJOR_TOPICS, SEED

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUTD = os.path.join(ROOT, "results", "annotations")
os.makedirs(OUTD, exist_ok=True)

# Six instruction-tuned models from five independent developers, 1B to 3.8B.
# Llama and Gemma are gated behind an access request and were excluded so the whole
# pipeline stays reproducible by any reader without credentials -- the same reason the
# study uses local models at all. Every model is loaded in 4-bit NF4: forced by the
# 4 GB card, but also the cleaner comparison, since a difference between models cannot
# then be an artefact of one being quantised and another not.
MODELS = {
    "olmo2-1b":      ("allenai/OLMo-2-0425-1B-Instruct",       "nf4"),
    "qwen2.5-1.5b":  ("Qwen/Qwen2.5-1.5B-Instruct",            "nf4"),
    "smollm2-1.7b":  ("HuggingFaceTB/SmolLM2-1.7B-Instruct",   "nf4"),
    "granite3.1-2b": ("ibm-granite/granite-3.1-2b-instruct",   "nf4"),
    "qwen2.5-3b":    ("Qwen/Qwen2.5-3B-Instruct",              "nf4"),
    "phi3.5-mini":   ("microsoft/Phi-3.5-mini-instruct",       "nf4"),
}
CONDITIONS = ["zs", "cb", "fs"]          # zero-shot / codebook definitions / few-shot
NAMES = [CAP_MAJOR_TOPICS[c] for c in VALID_CODES]
K_SHOT = 5
FEWSHOT_TRUNC_WORDS = 60
BATCH = 4

# Short glosses used only in the "cb" condition; wording follows the CAP master codebook.
CODEBOOK_GLOSS = {
    1:  "inflation, prices, interest rates, unemployment rate, national budget and taxation in the aggregate",
    2:  "discrimination, voting rights, privacy, freedom of speech and other civil liberties",
    3:  "health care provision, insurance, disease, drug regulation, medical research",
    4:  "farming, agricultural trade and subsidies, food inspection, fisheries",
    5:  "worker safety, employee benefits, unions, training, migrant and youth labour",
    6:  "schools, higher education, vocational training, education funding and testing",
    7:  "pollution, waste, water and air quality, conservation of species, recycling",
    8:  "electricity, nuclear power, coal, oil and gas, alternative energy, conservation",
    9:  "immigration, refugees, citizenship and naturalisation",
    10: "roads, mass transit, aviation, railways, maritime transport, infrastructure safety",
    12: "crime, policing, courts, prisons, drug enforcement, family and domestic law",
    13: "poverty assistance, pensions, elderly and disabled care, social services, charity",
    14: "housing, urban and rural development, homelessness, community planning",
    15: "banking, securities, insurance, bankruptcy, consumer finance, small business, competition",
    16: "military personnel and procurement, defence alliances, arms control, veterans, intelligence",
    17: "space programmes, science funding, telecommunications, broadcasting, computing, weather",
    18: "trade agreements, tariffs, exports and imports, exchange rates, foreign investment",
    19: "foreign aid, diplomacy, international organisations, human rights abroad, developing world",
    20: "the machinery of government itself: agencies, civil service, elections, procurement, holidays and commemorations",
    21: "public lands, national parks, water resources, forests, territorial and indigenous land issues",
    23: "arts, culture, heritage, sport and recreation policy",
}

SYSTEM = ("You are an experienced content coder for the Comparative Agendas Project. "
          "You assign each text the single major policy topic that best describes its "
          "primary subject, following the CAP master codebook.")


def topic_block(with_gloss):
    if with_gloss:
        return "\n".join(f"- {n} — {CODEBOOK_GLOSS[c]}"
                         for n, c in zip(NAMES, VALID_CODES))
    return "\n".join(f"- {n}" for n in NAMES)


def build_prompt(tok, text, venue_desc, condition, shots):
    parts = [f"Assign the following text to exactly one CAP major topic.\n\n"
             f"Source: {venue_desc}\n\nTopics:\n{topic_block(condition == 'cb')}\n"]
    if condition == "fs" and shots:
        ex = []
        for s_text, s_name in shots:
            short = " ".join(s_text.split()[:FEWSHOT_TRUNC_WORDS])
            ex.append(f'Text: "{short}"\nAnswer: {s_name}')
        parts.append("Examples:\n" + "\n\n".join(ex) + "\n")
    parts.append(f'Text: "{text}"\n\nAnswer with the topic name exactly as written above.')
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "\n".join(parts)}]
    try:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        merged = [{"role": "user", "content": SYSTEM + "\n\n" + "\n".join(parts)}]
        return tok.apply_chat_template(merged, tokenize=False, add_generation_prompt=True)


def label_prefixes(tok):
    """Token prefix that identifies each topic, and the groups needing a second token.

    Returns (first_ids, ambiguous) where ambiguous maps a shared first token to
    {label index: second token id}. Five of the six tokenizers give 21 distinct first
    tokens and leave `ambiguous` empty.
    """
    seqs = [tok.encode(n, add_special_tokens=False) for n in NAMES]
    if any(len(s) == 0 for s in seqs):
        raise RuntimeError("a topic name tokenised to nothing")
    first = [s[0] for s in seqs]
    groups = {}
    for j, f in enumerate(first):
        groups.setdefault(f, []).append(j)
    ambiguous = {}
    for f, members in groups.items():
        if len(members) < 2:
            continue
        second = {}
        for j in members:
            if len(seqs[j]) < 2:
                raise RuntimeError(f"cannot separate topic {NAMES[j]!r} in this tokenizer")
            second[j] = seqs[j][1]
        if len(set(second.values())) != len(second):
            raise RuntimeError(f"topics collide beyond two tokens: "
                               f"{[NAMES[j] for j in members]}")
        ambiguous[f] = second
    return first, ambiguous


def load_corpus(key):
    with open(os.path.join(DATA, f"{key}.csv"), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


@torch.no_grad()
def score_prompts(model, tok, prompts, first_ids, ambiguous, dev):
    """Log-probability of each topic name at the answer position."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
    out = model(**enc, logits_to_keep=1)
    lp = torch.log_softmax(out.logits[:, -1, :].float(), dim=-1)
    idx = torch.tensor(first_ids, device=dev)
    scores = lp.index_select(1, idx).clone()             # (B, 21)
    for shared_tok, second in ambiguous.items():
        # both labels start with the same token, so the first-token score is identical
        # for them and cannot separate the pair; one extra pass reads the next token
        extra = torch.full((len(prompts), 1), shared_tok, device=dev, dtype=torch.long)
        enc2 = {"input_ids": torch.cat([enc["input_ids"], extra], dim=1),
                "attention_mask": torch.cat(
                    [enc["attention_mask"], torch.ones_like(extra)], dim=1)}
        lp2 = torch.log_softmax(model(**enc2, logits_to_keep=1).logits[:, -1, :].float(), dim=-1)
        for j, tok2 in second.items():
            scores[:, j] = scores[:, j] + lp2[:, tok2]
    return torch.softmax(scores, dim=-1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--conditions", nargs="*", default=CONDITIONS)
    ap.add_argument("--corpora", nargs="*", default=list(CORPORA))
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N items")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for mkey in args.models:
        repo, precision = MODELS[mkey]
        todo = [(cond, ckey) for cond in args.conditions for ckey in args.corpora
                if not os.path.exists(os.path.join(OUTD, f"{mkey}__{cond}__{ckey}.npz"))]
        if not todo:
            print(f"[{mkey}] tum shardlar hazir, atlaniyor", flush=True); continue

        print(f"\n=== {mkey} ({repo}, {precision}) yukleniyor -> {dev}", flush=True)
        tok = AutoTokenizer.from_pretrained(repo)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        first_ids, ambiguous = label_prefixes(tok)
        if ambiguous:
            print(f"  not: {sum(len(v) for v in ambiguous.values())} konu adi ayni token'la "
                  f"basliyor, ikinci token ile ayriliyor", flush=True)
        kw = {"device_map": dev, "attn_implementation": "eager"}
        if precision == "nf4" and dev == "cuda":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
        else:
            kw["dtype"] = torch.float16 if dev == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(repo, **kw)
        model.eval()

        for cond, ckey in todo:
            rows = load_corpus(ckey)
            if args.limit:
                rows = rows[:args.limit]
            venue_desc = CORPORA[ckey][5]

            shots = []
            if cond == "fs":
                rng = np.random.default_rng(SEED)
                by_lab = {}
                for r in rows:
                    if r["split"] == "calib":
                        by_lab.setdefault(int(r["label"]), []).append(r)
                name_of = {c: n for c, n in zip(VALID_CODES, NAMES)}
                for lab in rng.permutation(sorted(by_lab))[:K_SHOT]:
                    shots.append((by_lab[int(lab)][0]["text"], name_of[int(lab)]))

            probs = np.zeros((len(rows), len(VALID_CODES)), dtype=np.float32)
            t0 = time.time()
            order = np.argsort([len(r["text"]) for r in rows])   # length-bucketed batching
            batch = args.batch
            bstart = 0
            while bstart < len(order):
                idx = order[bstart:bstart + batch]
                prompts = [build_prompt(tok, rows[i]["text"], venue_desc, cond, shots)
                           for i in idx]
                try:
                    p = score_prompts(model, tok, prompts, first_ids, ambiguous, dev)
                except torch.OutOfMemoryError:
                    # long items (Supreme Court summaries run to ~250 tokens, and the
                    # few-shot condition adds five more) can still overflow. Halve the
                    # batch and retry the same items rather than losing the shard.
                    gc.collect(); torch.cuda.empty_cache()
                    if batch == 1:
                        raise
                    batch = max(1, batch // 2)
                    print(f"  [{mkey}|{cond}|{ckey}] OOM -> parti {batch}", flush=True)
                    continue
                for j, i in enumerate(idx):
                    probs[i] = p[j]
                bstart += len(idx)
                if bstart % (args.batch * 50) < batch:
                    rate = bstart / max(time.time() - t0, 1e-6)
                    print(f"  [{mkey}|{cond}|{ckey}] {bstart}/{len(rows)} "
                          f"({rate:.1f} oge/s, kalan ~{(len(rows)-bstart)/max(rate,1e-6)/60:.1f} dk)",
                          flush=True)

            gold = np.array([int(r["label"]) for r in rows], dtype=np.int16)
            np.savez_compressed(
                os.path.join(OUTD, f"{mkey}__{cond}__{ckey}.npz"),
                probs=probs, gold=gold,
                split=np.array([r["split"] for r in rows]),
                ids=np.array([r["id"] for r in rows]),
                year=np.array([int(r["year"]) for r in rows], dtype=np.int32),
                n_words=np.array([int(r["n_words"]) for r in rows], dtype=np.int32),
                codes=np.array(VALID_CODES, dtype=np.int16),
                meta=json.dumps({"model": mkey, "repo": repo, "precision": precision,
                                 "condition": cond, "corpus": ckey,
                                 "k_shot": K_SHOT if cond == "fs" else 0,
                                 "response_format": "topic-name first-token scoring",
                                 "seed": SEED, "n": len(rows),
                                 "elapsed_s": round(time.time() - t0, 1)}))
            acc = float((np.array(VALID_CODES)[probs.argmax(1)] == gold).mean())
            print(f"  -> {mkey}|{cond}|{ckey}: acc={acc:.3f}  "
                  f"({time.time()-t0:.0f} s) kaydedildi", flush=True)

        del model
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
