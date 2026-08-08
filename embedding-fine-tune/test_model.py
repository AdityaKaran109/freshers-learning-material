"""
test_model.py
-------------
Sanity-check an embedding model and visualise how it groups drug brands.

The brand list below is taken from the A-Z Medicine Dataset of India and
labelled with each brand's REAL composition (short_composition1/2), not a
loose "drug class". That distinction matters:

    Augmentin 625 Duo = Amoxicillin + Clavulanic Acid, NOT plain Amoxicillin
    Combiflam         = Ibuprofen + Paracetamol,       NOT plain Ibuprofen

A good model should put Augmentin near Clavam and AWAY from Mox, even though
all of them contain amoxicillin. The single-salt vs combination pairs below
are the interesting test; anything can cluster paracetamol against antibiotics.

Read the NUMBERS, not the picture. PCA is refit per model, so axes are
arbitrary and positions are not comparable between two runs.

Run:
    python test_model.py --model ./models/drug-embed-v1 --title "After fine-tuning" --out after.png
    python test_model.py --model ./FremyCompany/BioLORD-2023 --title "Before fine-tuning" --out before.png
"""

import argparse
import itertools

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# composition -> brands that really have that composition
TEST_SET = {
    "Paracetamol":                     ["Dolo 650", "Calpol 650mg", "Crocin Advance", "Pacimol 500"],
    "Ibuprofen":                       ["Brufen 400", "Ibugesic 400", "Emflam 400mg", "Avibru 400mg"],
    "Ibuprofen + Paracetamol":         ["Combiflam", "Flexon", "Imol Plus", "Ibugesic Plus"],
    "Amoxicillin":                     ["Mox 500mg", "Novamox 500", "Almox 500", "Wymox 500mg"],
    "Amoxicillin + Clavulanic Acid":   ["Augmentin 625 Duo", "Clavam 625", "Moxikind-CV 625", "Mox CV 625"],
    "Pantoprazole":                    ["PAN 40", "Pantop 40", "Pantocid", "Pentab 40"],
    "Atorvastatin":                    ["Atorva 20", "Storvas 20", "Tonact 20", "Aztor 20"],
}

# Pairs that look alike but must NOT collapse together: same brand family or
# same first salt, different actual composition.
LOOKALIKES = [
    ("Mox 500mg", "Novamox 500", "same"),
    ("Mox 500mg", "Mox CV 625", "different"),
    ("Augmentin 625 Duo", "Clavam 625", "same"),
    ("Augmentin 625 Duo", "Novamox 500", "different"),
    ("Brufen 400", "Ibugesic 400", "same"),
    ("Ibugesic 400", "Ibugesic Plus", "different"),
    ("Combiflam", "Flexon", "same"),
    ("Combiflam", "Brufen 400", "different"),
    ("Dolo 650", "Crocin Advance", "same"),
    ("Dolo 650", "Combiflam", "different"),
]


def load_model(model_dir):
    print(f"Loading {model_dir} on {DEVICE}")
    return SentenceTransformer(model_dir, device=str(DEVICE))


def embed(model, texts):
    """Turn a list of strings into normalized embedding vectors."""
    return model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def smoke_test(model):
    print("\n--- MODEL SMOKE TEST ---")
    v = embed(model, ["hello world"])
    print("Embedding shape :", v.shape)
    print("First 5 numbers :", np.round(v[0, :5], 4))
    print("Model params    :", sum(p.numel() for p in model.parameters()))
    print("MODEL OK (loads, runs a forward pass, returns embeddings)")


def report(model, test_set=TEST_SET):
    """Numbers that actually say whether the model learned composition."""
    brands = [b for group in test_set.values() for b in group]
    labels = [label for label, group in test_set.items() for _ in group]
    vecs = embed(model, brands)
    sim = cosine_similarity(vecs)
    n = len(brands)

    same = [sim[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] == labels[j]]
    diff = [sim[i, j] for i, j in itertools.combinations(range(n), 2) if labels[i] != labels[j]]

    # Nearest neighbour (excluding self): does it share the composition?
    np.fill_diagonal(sim, -np.inf)
    nn = sim.argmax(axis=1)
    hits = [labels[i] == labels[nn[i]] for i in range(n)]

    print("\n--- COMPOSITION CLUSTERING ---")
    print(f"brands: {n}   compositions: {len(test_set)}")
    print(f"mean cosine, same composition : {np.mean(same):.4f}")
    print(f"mean cosine, diff composition : {np.mean(diff):.4f}")
    print(f"separation margin             : {np.mean(same) - np.mean(diff):+.4f}")
    print(f"nearest-neighbour accuracy    : {np.mean(hits):.4f}  ({sum(hits)}/{n})")

    misses = [(brands[i], labels[i], brands[nn[i]], labels[nn[i]])
              for i in range(n) if not hits[i]]
    if misses:
        print("  misgrouped (nearest neighbour has a different composition):")
        for b, lb, o, lo in misses:
            print(f"    {b:20} [{lb}]  ->  {o} [{lo}]")

    print("\n--- LOOK-ALIKE PAIRS (the real test) ---")
    print(f"  {'pair':46} {'truth':10} cosine")
    index = {b: i for i, b in enumerate(brands)}
    for a, b, truth in LOOKALIKES:
        s = float(cosine_similarity([vecs[index[a]]], [vecs[index[b]]])[0, 0])
        print(f"  {a + '  vs  ' + b:46} {truth:10} {s:.4f}")

    return {
        "same": float(np.mean(same)),
        "diff": float(np.mean(diff)),
        "margin": float(np.mean(same) - np.mean(diff)),
        "nn_accuracy": float(np.mean(hits)),
    }


def plot(model, test_set=TEST_SET, title="Embeddings", out="plot.png", stats=None):
    """Embed brands, project to 2D with PCA, and save a scatter plot."""
    brands = [b for group in test_set.values() for b in group]
    labels = [label for label, group in test_set.items() for _ in group]

    coords = PCA(n_components=2).fit_transform(embed(model, brands))

    unique = list(test_set)
    palette = plt.get_cmap("tab10").colors
    color_map = {label: palette[i % len(palette)] for i, label in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(11, 7.5))
    for label in unique:
        idx = [i for i, l in enumerate(labels) if l == label]
        ax.scatter(coords[idx, 0], coords[idx, 1],
                   color=color_map[label], s=90, label=label,
                   edgecolors="white", linewidths=0.6, zorder=3)
    for i, brand in enumerate(brands):
        ax.annotate(brand, (coords[i, 0], coords[i, 1]), xytext=(5, 4),
                    textcoords="offset points", fontsize=8, zorder=4)

    if stats:
        ax.set_title(f"{title}\nsame-composition cosine {stats['same']:.3f} vs "
                     f"different {stats['diff']:.3f}  "
                     f"(margin {stats['margin']:+.3f}, NN acc {stats['nn_accuracy']:.2f})",
                     fontsize=11)
    else:
        ax.set_title(title)

    ax.set_xlabel("PC 1 (arbitrary — refit per model, not comparable across plots)")
    ax.set_ylabel("PC 2 (arbitrary)")
    ax.legend(title="Actual composition", fontsize=8, title_fontsize=9, loc="best")
    ax.grid(alpha=0.15, zorder=0)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="./models/drug-embed-v1")
    ap.add_argument("--title", default="After fine-tuning")
    ap.add_argument("--out", default="after.png")
    args = ap.parse_args()

    model = load_model(args.model)
    smoke_test(model)
    stats = report(model)
    plot(model, title=args.title, out=args.out, stats=stats)
