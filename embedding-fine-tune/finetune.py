"""
finetune.py
-----------
Fine-tune an embedding model so that drug brands with the SAME active
ingredient sit close together, and look-alike different drugs sit apart.

Reads the files produced by prepare_data.py:
    data/train_pairs.csv      (text_a, text_b)              -> positives
    data/train_triplets.csv   (anchor, positive, hard_negative)

Recipe:
    MultipleNegativesRankingLoss (MNRL) — the standard, strong loss for
    retrieval/matching. Every positive pair uses the rest of the batch as
    negatives; the triplet file adds EXPLICIT hard negatives on top.

Run:
    python finetune.py --base cambridgeltl/SapBERT-from-PubMedBERT-fulltext \
                       --data ./data --out ./models/drug-embed-v1

Requires: pip install "sentence-transformers>=3.0" torch pandas
"""

import argparse
import random

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.evaluation import TripletEvaluator

random.seed(42)


def load_pairs(path):
    df = pd.read_csv(path).dropna()
    return [InputExample(texts=[str(a), str(b)]) for a, b in zip(df.text_a, df.text_b)]


def load_triplets(path):
    df = pd.read_csv(path).dropna()
    return [InputExample(texts=[str(a), str(p), str(n)])
            for a, p, n in zip(df.anchor, df.positive, df.hard_negative)]


def triplet_accuracy(evaluator, model):
    """TripletEvaluator returns a dict in sentence-transformers v3+."""
    scores = evaluator(model)
    if isinstance(scores, dict):
        for key, value in scores.items():
            if key.endswith("_cosine_accuracy"):
                return float(value)
        return float(next(iter(scores.values())))
    return float(scores)


def main(base, data_dir, out_dir, epochs, batch_size, lr):
    # ---- 1. load base model on the NVIDIA GPU ----
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Install a CUDA-enabled PyTorch build before training."
        )

    device = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    print(f"Loading base model: {base}")
    model = SentenceTransformer(base, device=device)
    print(f"Model device: {model.device}")

    # ---- 2. load training data ----
    pairs = load_pairs(f"{data_dir}/train_pairs.csv")
    triplets = load_triplets(f"{data_dir}/train_triplets.csv")
    print(f"Loaded {len(pairs)} positive pairs, {len(triplets)} triplets")

    # hold out some triplets to MEASURE quality (is positive closer than hard-neg?)
    random.shuffle(triplets)
    n_eval = max(1, min(300, int(len(triplets) * 0.2)))
    eval_trip, train_trip = triplets[:n_eval], triplets[n_eval:]

    evaluator = TripletEvaluator(
        anchors=[e.texts[0] for e in eval_trip],
        positives=[e.texts[1] for e in eval_trip],
        negatives=[e.texts[2] for e in eval_trip],
        name="drug-val",
    )

    # ---- 3. baseline BEFORE training (this is your "before" number) ----
    print("\n--- BASELINE (before fine-tuning) ---")
    base_score = triplet_accuracy(evaluator, model)
    print(f"Triplet accuracy before: {base_score:.4f}")

    # ---- 4. train ----
    loss = losses.MultipleNegativesRankingLoss(model)
    train_objectives = [(DataLoader(pairs, shuffle=True, batch_size=batch_size), loss)]
    if train_trip:  # add hard-negative objective if we have triplets
        train_objectives.append(
            (DataLoader(train_trip, shuffle=True, batch_size=batch_size), loss)
        )

    warmup = int(len(pairs) / batch_size * epochs * 0.1)
    print(f"\nTraining: {epochs} epochs, batch {batch_size}, warmup {warmup} steps")
    model.fit(
        train_objectives=train_objectives,
        evaluator=evaluator,
        epochs=epochs,
        warmup_steps=warmup,
        optimizer_params={"lr": lr},
        output_path=out_dir,
        save_best_model=True,
        show_progress_bar=True,
    )

    # ---- 5. score AFTER training (reload best model) ----
    print("\n--- RESULT (after fine-tuning) ---")
    best = SentenceTransformer(out_dir)
    after_score = triplet_accuracy(evaluator, best)
    print(f"Triplet accuracy before: {base_score:.4f}")
    print(f"Triplet accuracy after : {after_score:.4f}")
    print(f"Model saved to: {out_dir}")
    print("\nNow re-run test_and_plot.py with MODEL_DIR set to this folder "
          "to get your 'after' graph.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
                    help="base model id or local path")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./models/drug-embed-v1")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()
    main(args.base, args.data, args.out, args.epochs, args.batch_size, args.lr)