"""
Leakage audit for a finished evaluation run.

WHY THIS EXISTS
---------------
A suspiciously good WER is a bug report, not a result. This script is the
evidence behind the leakage claim in your model card.

It cannot verify speaker-disjointness -- Vaani has no speaker ID column, so
nobody downstream can. What it CAN do is rule out the failure modes that are
detectable from text alone:

  1. Verbatim transcript overlap between train and test
  2. Whether removing those overlaps moves the score (the decisive test)
  3. Whether perfect-scoring utterances are disproportionately memorised text
  4. Whether any utterance-length bucket is anomalously easy
  5. Whether the error distribution has the bimodal shape leakage produces

If duplicates are rare AND removing them barely moves WER AND the model does
no better on them than on anything else, text leakage is not inflating your
number.

Usage:
    python check_leakage.py --data data/bhojpuri --dump tuned_small.jsonl
"""

import argparse
import json
import statistics
from collections import Counter

from datasets import load_from_disk
from jiwer import cer, wer


def corpus(rows):
    """Corpus-level WER/CER over a list of {ref, hyp} rows."""
    if not rows:
        return None, None
    refs = [r["ref"] for r in rows]
    hyps = [r["hyp"] for r in rows]
    return round(100 * wer(refs, hyps), 2), round(100 * cer(refs, hyps), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bhojpuri",
                    help="Folder written by prepare_data.py")
    ap.add_argument("--dump", default="tuned_small.jsonl",
                    help="Per-utterance JSONL written by evaluate.py --dump")
    ap.add_argument("--split", default="test",
                    help="Which split --dump was produced from")
    args = ap.parse_args()

    ds = load_from_disk(args.data)
    train_texts = Counter(ds["train"]["text"])
    train_set = set(train_texts)
    print(f"train utterances : {len(ds['train'])} "
          f"({len(train_set)} unique transcripts)")

    rows = [json.loads(l) for l in open(args.dump, encoding="utf-8")]
    print(f"{args.split} utterances  : {len(rows)}\n")

    dup = [r for r in rows if r["ref"] in train_set]
    uniq = [r for r in rows if r["ref"] not in train_set]
    pct = 100 * len(dup) / len(rows)

    print("=" * 66)
    print("1. VERBATIM TRANSCRIPT OVERLAP")
    print("=" * 66)
    print(f"  transcripts also in train : {len(dup)}/{len(rows)}  ({pct:.2f}%)")
    print(f"  transcripts unique        : {len(uniq)}/{len(rows)}  ({100-pct:.2f}%)")

    all_w, all_c = corpus(rows)
    dup_w, dup_c = corpus(dup)
    un_w, un_c = corpus(uniq)

    print("\n" + "=" * 66)
    print("2. THE DECISIVE NUMBER -- score with duplicates REMOVED")
    print("=" * 66)
    print(f"  full split             : WER {all_w}   CER {all_c}")
    if dup:
        print(f"  duplicated transcripts : WER {dup_w}   CER {dup_c}  (n={len(dup)})")
        print("     ^ if this is much BETTER than the corpus average, the model")
        print("       is recalling training text rather than transcribing audio")
    print(f"  duplicates EXCLUDED    : WER {un_w}   CER {un_c}  (n={len(uniq)})")
    if un_w is not None:
        print(f"\n  shift from removing duplicates: {un_w - all_w:+.2f} WER")
        print("  A large POSITIVE shift means duplicates were inflating the score.")
        print("  Near zero means they were not.")

    print("\n" + "=" * 66)
    print("3. PERFECT SCORES -- disproportionately memorised text?")
    print("=" * 66)
    perfect = [r for r in rows if r["wer"] == 0]
    perfect_dup = [r for r in perfect if r["ref"] in train_set]
    print(f"  utterances at WER 0        : {len(perfect)}/{len(rows)} "
          f"({100*len(perfect)/len(rows):.1f}%)")
    if perfect:
        share = 100 * len(perfect_dup) / len(perfect)
        print(f"  of those, verbatim in train: {len(perfect_dup)}/{len(perfect)} "
              f"({share:.1f}%)")
        print(f"  base rate of duplicates    : {pct:.1f}%")
        if pct > 0:
            print(f"  enrichment: {share / pct:.2f}x  "
                  "(1.0 = no relationship, >2 = suspicious)")
        if len(perfect_dup) < 5:
            print("  NOTE: fewer than 5 overlapping hits -- the enrichment ratio is")
            print("        statistical noise at this count. Do not read into it.")

    print("\n" + "=" * 66)
    print("4. SCORE BY REFERENCE LENGTH (an easy bucket = memorisation)")
    print("=" * 66)
    for lo, hi in [(1, 3), (4, 6), (7, 10), (11, 20), (21, 10**6)]:
        sub = [r for r in rows if lo <= len(r["ref"].split()) <= hi]
        if sub:
            w, c = corpus(sub)
            label = f"{lo}-{hi}" if hi != 10**6 else f"{lo}+"
            flag = "   <- n too small to interpret" if len(sub) < 10 else ""
            print(f"  {label:>6} words : n={len(sub):>4}   WER {w:>6}   "
                  f"CER {c:>6}{flag}")

    print("\n" + "=" * 66)
    print("5. ERROR DISTRIBUTION (leakage is bimodal: many 0s, many disasters)")
    print("=" * 66)
    ws = sorted(r["wer"] for r in rows)
    print(f"  mean {100*statistics.mean(ws):.1f}   "
          f"median {100*statistics.median(ws):.1f}")
    for q, name in [(0.25, "p25"), (0.50, "p50"), (0.75, "p75"), (0.90, "p90")]:
        print(f"  {name}: {100*ws[int(q*len(ws))]:.1f}")
    over = sum(1 for w in ws if w > 1.0)
    print(f"  utterances above 100% WER: {over} ({100*over/len(ws):.1f}%) "
          "-- hallucination loops")

    print("\n" + "=" * 66)
    print("VERDICT")
    print("=" * 66)
    clean = (pct < 2.0 and un_w is not None and abs(un_w - all_w) < 1.0
             and (not dup or dup_w >= all_w - 5))
    if clean:
        print("  No text leakage detected. Duplicates are rare, removing them does")
        print("  not move the score, and the model does not do unusually well on")
        print("  them.")
    else:
        print("  Something here needs explaining before you publish. Check which of")
        print("  sections 1-3 tripped.")
    print("\n  NOT PROVEN: speaker-disjointness. This dataset has no speaker ID")
    print("  column, so that is trusted from the official splits, never verified.")
    print("  Say so explicitly in your model card.")


if __name__ == "__main__":
    main()
