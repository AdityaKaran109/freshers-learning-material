# Fine-tuning an embedding model on drug brand names

Teach a sentence-embedding model that two medicine brands with the **same active
ingredients** belong together — and that two brands whose *names* look alike do
not.

```
Mox 500mg      = Amoxicillin                     \  same molecule,
Novamox 500    = Amoxicillin                     /  should be CLOSE

Ibugesic 400   = Ibuprofen                       \  near-identical names,
Ibugesic Plus  = Ibuprofen + Paracetamol         /  should be FAR APART
```

The second case is the entire difficulty. Measured on the base model
`BioLORD-2023`, before any training:

```
Ibugesic 400 vs Ibugesic Plus   different composition   0.9656   <- nearly identical
Brufen 400   vs Ibugesic 400    same composition        0.6821   <- actually the same drug
```

It ranks a *wrong* answer far above a *right* one. Across 15,575 such look-alike
triplets the base model picks the correct one **16.9%** of the time — chance is
50%, so it is not merely ignorant, it is reliably pulled the wrong way by shared
spelling. Fixing that is what this project is about.

## What's in here

Code **and** the prepared training data. Not included: model weights,
checkpoints, and the 32 MB raw Kaggle export — `model_download.py` fetches the
last one in seconds, so committing it would only bloat the repo.

```
model_download.py    base model + dataset download          (run 1st)
prepare_dataset.py   dataset -> pairs and triplets          (run 2nd, optional)
finetune.py          the fine-tune                          (run 3rd, needs a CUDA GPU)
test_model.py        metrics + PCA plot                     (run 4th)
data/                prepared training data, ready to use
requirements.txt
```

Step 2 is optional **because `data/` is already committed** — you can go
straight from download to training. Re-run it if you want to change how the
data is built.

| `data/` file | rows | what it is |
|---|---|---|
| `train_pairs.csv` | 371,947 | `text_a, text_b` — same-composition positives |
| `train_triplets.csv` | 218,643 | `anchor, positive, hard_negative` |
| `val_pairs.csv` | 37,002 | held-out brands |
| `val_triplets.csv` | 23,729 | held-out brands — **primary metric** |
| `val_lookalike_triplets.csv` | 15,575 | name-collision negatives only — **the hard metric** |
| `unseen_comp_triplets.csv` | 25,211 | compositions never trained on — secondary |

Derived from the
[A-Z Medicine Dataset of India](https://www.kaggle.com/datasets/shudhanshusingh/az-medicine-dataset-of-india)
(253,973 medicines with real `short_composition1` / `short_composition2`
columns). These files are transformations of it, not a copy — check the
dataset's terms before reusing them commercially.

## Setup

```bash
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash; venv\Scripts\activate on cmd
                                  # source venv/bin/activate on Linux/macOS
pip install -r requirements.txt
```

Install PyTorch matched to your CUDA version — see the comment at the top of
`requirements.txt`. `finetune.py` requires a CUDA GPU and exits without one.

## 1. Download the base model

```bash
python model_download.py
```

Pulls [`FremyCompany/BioLORD-2023`](https://huggingface.co/FremyCompany/BioLORD-2023)
— a biomedical sentence-embedding model, a better start than a general-purpose
one — plus the raw dataset. Both are skipped if already present; no Kaggle
credentials needed.

## 2. Build the training data *(optional — `data/` is committed)*

```bash
python prepare_dataset.py \
    --csv "shudhanshusingh/az-medicine-dataset-of-india/A_Z_medicines_dataset_of_India.csv" \
    --out ./data
```

Three decisions in here matter more than anything in the training script.

**Split by brand, not by composition.** Holding out whole compositions asks the
model to recognise a molecule it has never seen named even once — a brand name
carries no signal for that. Worse, it silently removes those molecules from
training: an earlier version of this project held out paracetamol, amoxicillin
and atorvastatin, then evaluated on `Dolo 650` and `Mox 500mg` and concluded the
model had failed. All 2,379 compositions now train; 23,412 *brands* are held out
inside them, which is the real task — a new brand launches for a known salt. A
separate 118 compositions are withheld as a deliberately hard secondary metric.

**Mine look-alike negatives.** Random negatives teach almost nothing. Two
sources of genuinely hard ones:

- *root collisions* — the same brand family across different compositions
  (`Ibugesic` / `Ibugesic Plus`, `Mox` / `Mox CV`): 2,675 roots, 4,692 triplets
- *prefix collisions* — a shared 5-character opening, different composition
  (`Adyom D 30mg/40mg` / `Adyom LS 75mg/40mg`): 86,765 triplets

Root matching alone misses `Mox 500mg` vs `Moxikind-CV 625`; the prefix index
catches it. Coincidental prefix collisions are fine — the pair still looks alike
to a tokenizer, which is exactly what must be told apart.

**Ground every brand to its composition text.** Pairing `Mox 500mg` with
`Amoxicillin` gives a target the biomedical base model already understands,
instead of only shoving arbitrary brand strings toward each other.

Smaller things that matter: canonical salt keys (`Amoxycillin (500mg)` →
`amoxicillin`, order-independent sets), alias collapsing
(`Paracetamol/Acetaminophen`), counter-ion stripping (`Diclofenac Sodium` →
`diclofenac`), dosage-form trimming (`Augmentin 625 Duo Tablet` → `Augmentin 625
Duo`), bare-brand recovery (`Calpol 500mg` → `Calpol`, only when unambiguous),
and dropping ~6.2k rows whose title implies more salts than the two recorded
columns can hold.

Groups are huge — 5,883 brands share *amoxicillin + clavulanic acid*.
Enumerating all pairs is `C(n,2)` and exhausts memory; sampling a flat 20 per
group leaves most brands unused. Each group is shuffled and neighbours paired,
so every brand appears without ever materialising all combinations.

## 3. Fine-tune

```bash
python finetune.py --base ./FremyCompany/BioLORD-2023 --out ./models/drug-embed-v2
```

Uses **`CachedGISTEmbedLoss`**, not plain `MultipleNegativesRankingLoss`.

MNRL treats every other item in the batch as a negative. With 5,883 brands
sharing one composition, a random batch routinely holds two brands of the *same*
composition — and MNRL then teaches the model they are unrelated. Those false
negatives compress the space instead of separating it. GIST uses a frozen guide
model to drop in-batch negatives that look more similar than the true positive,
which removes the failure mode; the `Cached` variant adds gradient caching so
batch size is bounded by the dataset rather than by VRAM (512 here).

## 4. Evaluate

```bash
python test_model.py --model ./models/drug-embed-v2 --title "After fine-tuning" --out after.png
python test_model.py --model ./FremyCompany/BioLORD-2023 --title "Before fine-tuning" --out before.png
```

28 real brands across 7 verified compositions, including single-salt vs
combination pairs of the same molecule. Separating paracetamol from antibiotics
is easy; keeping `Ibugesic 400` away from `Ibugesic Plus` is the job.

## Reading the results honestly

Three traps, each of which produced a convincing-looking but meaningless result
earlier in this project.

**Do not compare PCA plots between two models.** PCA is refit on each set of
embeddings, so the axes are arbitrary — rotations, flips and shifts between a
"before" and "after" plot mean nothing. Compare the printed metrics.

**Check your labels before believing a failure.** `Augmentin` is amoxicillin
**+ clavulanic acid**; `Combiflam` is ibuprofen **+ paracetamol**. A model that
separates them from plain amoxicillin and plain ibuprofen is being *right*, even
though a naive "drug class" label scores it as wrong.

**A rising score can hide a worse model, and a falling one can hide a better
one.** Fine-tuning inflates *every* cosine at once. Same-composition similarity
climbs, which looks like success, but different-composition climbs with it — so
the raw margin between them shrinks even as the ranking improves. Ranking is
what retrieval depends on, so judge on rank-based metrics (nearest-neighbour
accuracy, triplet accuracy) and use a normalized margin — the gap divided by the
spread of similarities — if you want a margin at all. This project selects
checkpoints on nearest-neighbour accuracy for exactly this reason.

The cost of getting that wrong, from a real run on this repo's data: raw margin
fell from 0.120 to 0.065 while normalized margin rose from 0.87 to 1.40 and
nearest-neighbour accuracy went 0.239 → 0.430. Selecting on raw margin would
have discarded the better model.

### Baseline to beat

`BioLORD-2023`, untrained, on the held-out sets in `data/`:

| metric | base model |
|---|---|
| composition nearest-neighbour accuracy | 0.2394 |
| held-out brand triplets | 0.3322 |
| **look-alike triplets** | **0.1685** |
| unseen-composition triplets | 0.2690 |
| normalized margin | 0.8667 |

Every triplet number is below the 0.50 chance line — the base model is actively
misled by name similarity. Note these are far harsher than the 0.9267 an earlier
version of this project reported, because that number came from *randomly*
sampled negatives and from labels inferred from product titles. Lower numbers
that you can trust beat higher ones you cannot.

## Known limitations

- Brand → composition is substantially a **memorization** task. For brands in
  the catalogue this works well; for a genuinely novel brand name only
  morphology helps (`-cef` → cephalosporin, `CV` → clavulanate), so expect
  limited generalization there. The `unseen_comp_triplets.csv` score is the
  honest read on that.
- The source stores at most two salts per medicine, so true three-ingredient
  products cannot be represented and are dropped.
- Indian brands, allopathy only. `Advil` is not in the dataset.
