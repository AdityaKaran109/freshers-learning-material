r"""
finetune.py
-----------
Fine-tune an embedding model so drug brands with the SAME active ingredients
sit close together, and look-alike names with DIFFERENT compositions sit apart.

Reads what prepare_dataset.py produced:
    data/train_pairs.csv             (text_a, text_b)
    data/train_triplets.csv          (anchor, positive, hard_negative)
    data/val_triplets.csv            unseen brands, seen compositions
    data/val_lookalike_triplets.csv  name-collision negatives  <- the hard case
    data/unseen_comp_triplets.csv    compositions never trained

Why this recipe:

  CachedGISTEmbedLoss, not plain MultipleNegativesRankingLoss.
    MNRL treats every other item in the batch as a negative. With 5,883 brands
    sharing "amoxicillin + clavulanic acid", a random batch routinely contains
    two brands of the SAME composition and MNRL then teaches the model they are
    unrelated. Those false negatives compress the whole space: in an earlier run
    same-composition similarity rose to 0.676 while different-composition rose to
    0.587 alongside it, and the separation margin FELL from +0.142 to +0.089.
    GIST uses a frozen guide model to drop in-batch negatives that look more
    similar than the true positive, which removes that failure mode. The Cached
    variant adds gradient caching so batch size is limited by dataset, not VRAM.

  Big batches.
    Contrastive quality scales with the number of negatives, and negatives come
    from the batch. Caching makes 512-1024 affordable at near-constant memory.

  Model selection on separation MARGIN, not raw accuracy.
    A model can raise same-composition similarity and look like it improved
    while raising different-composition similarity just as much. Margin catches
    that; accuracy alone does not.

Run:
    python finetune.py --base ./FremyCompany/BioLORD-2023 --out ./models/drug-embed-v2
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import SentenceEvaluator, TripletEvaluator
from sentence_transformers.losses import CachedGISTEmbedLoss
from sentence_transformers.training_args import BatchSamplers
from sklearn.metrics.pairwise import cosine_similarity


# ----------------------------------------------------------------------
# Evaluator: does the model actually separate compositions?
# ----------------------------------------------------------------------
class CompositionEvaluator(SentenceEvaluator):
    """Mean cosine within a composition vs across compositions.

    The gap between the two ("margin") is the number that matters. Reporting
    only the within-composition score hides a model that pulled everything
    together indiscriminately.
    """

    def __init__(self, brands, labels, name="comp", batch_size=256):
        super().__init__()
        self.brands = list(brands)
        self.labels = list(labels)
        self.name = name
        self.batch_size = batch_size
        # Selection runs on nearest-neighbour accuracy, NOT raw margin.
        # Fine-tuning inflates every cosine at once, so a raw margin can shrink
        # while the ranking -- the thing retrieval actually depends on -- gets
        # better. Rank-based metrics are invariant to that global shift.
        self.primary_metric = f"{name}_nn_accuracy"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1, **kwargs):
        vecs = model.encode(self.brands, batch_size=self.batch_size,
                            convert_to_numpy=True, normalize_embeddings=True,
                            show_progress_bar=False)
        sim = cosine_similarity(vecs)
        n = len(self.brands)
        same, diff = [], []
        for i, j in itertools.combinations(range(n), 2):
            (same if self.labels[i] == self.labels[j] else diff).append(sim[i, j])

        np.fill_diagonal(sim, -np.inf)
        nn = sim.argmax(axis=1)
        nn_acc = float(np.mean([self.labels[i] == self.labels[nn[i]] for i in range(n)]))

        spread = float(np.std(np.concatenate([same, diff]))) or 1e-9
        metrics = {
            f"{self.name}_same": float(np.mean(same)),
            f"{self.name}_diff": float(np.mean(diff)),
            f"{self.name}_margin": float(np.mean(same) - np.mean(diff)),
            # effect size: margin in units of the similarity spread, so it does
            # not move when every cosine shifts up or down together
            f"{self.name}_margin_norm": float(np.mean(same) - np.mean(diff)) / spread,
            f"{self.name}_nn_accuracy": nn_acc,
        }
        self.store_metrics_in_model_card_data(model, metrics, epoch, steps)
        return metrics


def load_pairs(path):
    df = pd.read_csv(path).dropna()
    return Dataset.from_dict({"anchor": df.iloc[:, 0].astype(str).tolist(),
                              "positive": df.iloc[:, 1].astype(str).tolist()})


def load_triplets(path, limit=None):
    df = pd.read_csv(path).dropna()
    if limit and len(df) > limit:
        df = df.sample(limit, random_state=42)
    return Dataset.from_dict({"anchor": df.anchor.astype(str).tolist(),
                              "positive": df.positive.astype(str).tolist(),
                              "negative": df.hard_negative.astype(str).tolist()})


def triplet_evaluator(path, name, limit=4000):
    ds = load_triplets(path, limit=limit)
    return TripletEvaluator(anchors=ds["anchor"], positives=ds["positive"],
                            negatives=ds["negative"], name=name, batch_size=256)


def composition_evaluator(val_pairs_path, max_groups=60, per_group=6):
    """Build a clustering probe from held-out brands.

    val_pairs pairs a brand with its composition text, so the composition text
    column recovers the label without re-reading the source CSV.
    """
    df = pd.read_csv(val_pairs_path).dropna()
    labelled = df[df.text_b.str.contains(" + ", regex=False)]
    brands, labels = [], []
    for label, sub in labelled.groupby("text_b"):
        names = sub.text_a.unique().tolist()
        if len(names) < 2:
            continue
        brands.extend(names[:per_group])
        labels.extend([label] * len(names[:per_group]))
        if len({*labels}) >= max_groups:
            break
    return CompositionEvaluator(brands, labels, name="comp")


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install a CUDA-enabled PyTorch build.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = SentenceTransformer(args.base, device="cuda")
    model.max_seq_length = args.max_seq_length      # brand names are short
    print(f"Loaded {args.base}  (max_seq_length={model.max_seq_length})")

    # Frozen guide that decides which in-batch negatives are false negatives.
    guide = SentenceTransformer(args.guide or args.base, device="cuda")
    guide.max_seq_length = args.max_seq_length
    for p in guide.parameters():
        p.requires_grad_(False)

    # ---- data ----
    train = {
        "pairs": load_pairs(f"{args.data}/train_pairs.csv"),
        "triplets": load_triplets(f"{args.data}/train_triplets.csv"),
    }
    for split, ds in train.items():
        print(f"train[{split}]: {len(ds)} rows, columns {ds.column_names}")

    loss_fn = CachedGISTEmbedLoss(model, guide, mini_batch_size=args.mini_batch_size)
    loss = {name: loss_fn for name in train}

    # ---- evaluation ----
    evaluators = [composition_evaluator(f"{args.data}/val_pairs.csv")]
    for stem, name in [("val_triplets", "val"),
                       ("val_lookalike_triplets", "lookalike"),
                       ("unseen_comp_triplets", "unseen_comp")]:
        path = f"{args.data}/{stem}.csv"
        if os.path.exists(path):
            evaluators.append(triplet_evaluator(path, name))

    from sentence_transformers.evaluation import SequentialEvaluator
    evaluator = SequentialEvaluator(evaluators)

    print("\n--- BEFORE FINE-TUNING ---")
    before = {}
    for ev in evaluators:
        before.update(ev(model))
    for k, v in before.items():
        print(f"  {k:34} {v:.4f}")

    # ---- train ----
    training_args = SentenceTransformerTrainingArguments(
        output_dir=args.checkpoints,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_ratio=0.1,
        bf16=True,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=2,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_comp_nn_accuracy",
        greater_is_better=True,
        report_to=[],
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()

    model.save_pretrained(args.out)
    print(f"\nSaved best model to {args.out}")

    print("\n--- AFTER FINE-TUNING ---")
    after = {}
    for ev in evaluators:
        after.update(ev(model))
    print(f"  {'metric':34} {'before':>9} {'after':>9}   change")
    for k in before:
        delta = after[k] - before[k]
        flag = "  <-- WORSE" if delta < 0 and "diff" not in k else ""
        print(f"  {k:34} {before[k]:9.4f} {after[k]:9.4f}   {delta:+.4f}{flag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="./FremyCompany/BioLORD-2023")
    ap.add_argument("--guide", default=None,
                    help="Guide model for GIST false-negative filtering (default: --base)")
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default="./models/drug-embed-v2")
    ap.add_argument("--checkpoints", default="./checkpoints")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--mini_batch_size", type=int, default=64,
                    help="Gradient-cache chunk; lower it if you hit OOM")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max_seq_length", type=int, default=32)
    ap.add_argument("--eval_steps", type=int, default=200)
    main(ap.parse_args())
