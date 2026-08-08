r"""
eval_retrieval.py
-----------------
Ask the question a product actually asks: given a brand name, WHAT IS IT?

Brand-vs-brand triplet accuracy is a proxy. The real task is retrieval against
the catalogue of compositions: embed "Mox 500mg", rank all ~2.4k composition
texts, and check "Amoxicillin" comes out on top.

It also separates two very different abilities, which the triplet metrics mix
together:

    SEEN brands   -- the model met this name in training. Getting these right
                     is memorisation, and for catalogue lookup that is exactly
                     what you want.
    UNSEEN brands -- the model has never met this name. Only morphology can
                     help (-cef -> cephalosporin, CV -> clavulanate), so this
                     is the honest generalization number and will be lower.

Reporting one number without the split hides which one you are buying.

Run:
    python eval_retrieval.py --model ./models/drug-embed-v2
    python eval_retrieval.py --model ./FremyCompany/BioLORD-2023   # baseline
"""

import argparse
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

import prepare_dataset as P

RAW_CSV = "shudhanshusingh/az-medicine-dataset-of-india/A_Z_medicines_dataset_of_India.csv"


def build_index(csv_path, min_group_size=2):
    """brand -> composition key, plus the list of candidate compositions."""
    df = pd.read_csv(csv_path)
    df = df.assign(
        comp_key=[P.composition_key(a, b)
                  for a, b in zip(df.short_composition1, df.short_composition2)],
        clean_name=df.name.map(P.clean_name),
    )
    df = df[(df.comp_key != "") & df.clean_name.map(P.is_writable_text)]
    per = df.groupby("clean_name")["comp_key"].nunique()
    df = df[~df.clean_name.isin(set(per[per > 1].index))].drop_duplicates("clean_name")

    counts = df.comp_key.value_counts()
    keys = sorted(counts[counts >= min_group_size].index)
    df = df[df.comp_key.isin(set(keys))]
    return dict(zip(df.clean_name, df.comp_key)), keys


def evaluate(model, brands, truth, keys, key_vecs, batch_size=256):
    vecs = model.encode(brands, batch_size=batch_size, convert_to_numpy=True,
                        normalize_embeddings=True, show_progress_bar=False)
    sims = vecs @ key_vecs.T                       # both normalized -> cosine
    order = np.argsort(-sims, axis=1)
    gold = np.array([keys.index(t) for t in truth])

    rank = np.array([np.where(order[i] == gold[i])[0][0] for i in range(len(brands))])
    return {
        "top1": float(np.mean(rank == 0)),
        "top5": float(np.mean(rank < 5)),
        "top10": float(np.mean(rank < 10)),
        "mrr": float(np.mean(1.0 / (rank + 1))),
        "median_rank": float(np.median(rank) + 1),
        "n": len(brands),
    }


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(args.model, device=device)
    model.max_seq_length = 32

    name_to_key, keys = build_index(args.csv)
    print(f"catalogue: {len(keys)} compositions, {len(name_to_key)} brands")

    key_vecs = model.encode([P.comp_text(k) for k in keys], batch_size=256,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)

    # Brands the model trained on vs brands held out, recovered from the
    # prepared splits so the definition matches training exactly.
    train_texts = set(pd.read_csv(f"{args.data}/train_pairs.csv").dropna().text_a) | \
                  set(pd.read_csv(f"{args.data}/train_pairs.csv").dropna().text_b)
    val_texts = set(pd.read_csv(f"{args.data}/val_pairs.csv").dropna().text_a) | \
                set(pd.read_csv(f"{args.data}/val_pairs.csv").dropna().text_b)

    rng = random.Random(42)
    seen = sorted(b for b in name_to_key if b in train_texts)
    unseen = sorted(b for b in name_to_key if b in val_texts and b not in train_texts)
    seen = rng.sample(seen, min(args.sample, len(seen)))
    unseen = rng.sample(unseen, min(args.sample, len(unseen)))

    print(f"\n--- BRAND -> COMPOSITION RETRIEVAL  ({args.model}) ---")
    print(f"  {'split':22} {'n':>6} {'top1':>7} {'top5':>7} {'top10':>7} {'MRR':>7} {'med rank':>9}")
    for label, brands in [("SEEN (memorisation)", seen), ("UNSEEN (generalization)", unseen)]:
        if not brands:
            continue
        m = evaluate(model, brands, [name_to_key[b] for b in brands], keys, key_vecs)
        print(f"  {label:22} {m['n']:>6} {m['top1']:>7.4f} {m['top5']:>7.4f} "
              f"{m['top10']:>7.4f} {m['mrr']:>7.4f} {m['median_rank']:>9.0f}")
    print(f"\n  random-guess top1 would be {1/len(keys):.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/drug-embed-v2")
    ap.add_argument("--csv", default=RAW_CSV)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--sample", type=int, default=3000)
    main(ap.parse_args())
