"""
prepare_dataset.py
------------------
Turn the A-Z Medicine Dataset of India into training data for embedding
fine-tuning.

Output teaches the model: "two brands with the SAME active-ingredient
composition should be close; a brand that only LOOKS similar (shares one salt
but not the full composition) should not."

Unlike the earlier 1mg dataset, this source has REAL composition columns
(short_composition1 / short_composition2), so nothing is guessed from titles.

Produces:
  train_pairs.csv / val_pairs.csv       -> (text_a, text_b) positive pairs
  train_triplets.csv / val_triplets.csv -> (anchor, positive, hard_negative)

Train and val use DISJOINT compositions, so val brands are unseen in training.

Run:
  python prepare_dataset.py \
      --csv "shudhanshusingh/az-medicine-dataset-of-india/A_Z_medicines_dataset_of_India.csv" \
      --out ./data
"""

import argparse
import itertools
import os
import random
import re
from collections import defaultdict

import pandas as pd

random.seed(42)

COLUMN_ALIASES = {
    "name": ["name", "medicine name", "medicine_name", "product name", "drug name"],
    "c1": [
        "short_composition1", "short composition1", "composition1",
        "salt composition", "compositions", "composition", "primary_composition",
    ],
    "c2": ["short_composition2", "short composition2", "composition2", "secondary_composition"],
}

# Collapse spelling variants so the same molecule gets one canonical key.
SPELL = {
    "amoxycillin": "amoxicillin",
    "acetaminophen": "paracetamol",
    "cefalexin": "cephalexin",
    "cefadroxyl": "cefadroxil",
    "salbutamol sulphate": "salbutamol",
    "frusemide": "furosemide",
    "rifampicin": "rifampin",
    "cholecalciferol": "vitamin d3",
    "ascorbic acid": "vitamin c",
    "thiamine": "vitamin b1",
    "pyridoxine": "vitamin b6",
    "cyanocobalamin": "vitamin b12",
    "methylcobalamin": "vitamin b12",
    "tocopherol": "vitamin e",
    "alpha tocopherol": "vitamin e",
}

# Counter-ions / hydrates: same active moiety, different packaging of it.
SALT_FORM_SUFFIXES = [
    "hydrochloride", "hcl", "dihydrochloride", "hydrobromide", "sodium",
    "potassium", "calcium", "magnesium", "sulphate", "sulfate", "maleate",
    "tartrate", "bitartrate", "citrate", "besylate", "mesylate", "fumarate",
    "succinate", "acetate", "phosphate", "nitrate", "oxalate", "gluconate",
    "lactate", "stearate", "palmitate", "dipropionate", "propionate",
    "valerate", "furoate", "monohydrate", "dihydrate", "trihydrate",
    "anhydrous", "micronized", "micronised",
]

# Trailing dosage-form words to strip off a product title.
DOSAGE_FORMS = [
    "tablet", "tablets", "tab", "tabs", "capsule", "capsules", "cap", "caps",
    "syrup", "suspension", "oral suspension", "oral solution", "solution",
    "injection", "infusion", "cream", "ointment", "gel", "lotion", "paint",
    "drop", "drops", "eye drop", "eye drops", "ear drop", "ear drops",
    "nasal drops", "nasal spray", "spray", "inhaler", "rotacaps", "respules",
    "powder", "granules", "sachet", "soap", "shampoo", "kit", "patch",
    "mouthwash", "gargle", "liquid", "emulsion", "lozenges", "suppository",
    "vaginal tablet", "transdermal patch", "pfs", "vial", "ampoule",
]
# Release modifiers that trail a dosage form ("Tablet SR", "Capsule ER").
RELEASE_MODS = ["sr", "er", "xr", "cr", "xl", "pr", "dr", "od", "dt", "md", "mr", "ir", "la"]

_FORM_RE = re.compile(
    r"\s*\b(?:" + "|".join(sorted(DOSAGE_FORMS, key=len, reverse=True)) + r")\b"
    r"(?:\s+\b(?:" + "|".join(RELEASE_MODS) + r")\b)?\s*$",
    flags=re.IGNORECASE,
)
_SALT_FORM_RE = re.compile(
    r"\s+(?:" + "|".join(sorted(SALT_FORM_SUFFIXES, key=len, reverse=True)) + r")$"
)


def resolve_col(df, aliases):
    """Match a dataframe column case-insensitively against known aliases."""
    lookup = {col.lower().strip(): col for col in df.columns}
    for alias in aliases:
        if alias in lookup:
            return lookup[alias]
    return None


def resolve_columns(df, name_col, c1, c2):
    resolved_name = name_col if name_col in df.columns else resolve_col(df, COLUMN_ALIASES["name"])
    resolved_c1 = c1 if c1 in df.columns else resolve_col(df, COLUMN_ALIASES["c1"])
    resolved_c2 = c2 if c2 in df.columns else resolve_col(df, COLUMN_ALIASES["c2"])

    if resolved_name is None:
        raise KeyError(
            f"Could not find a medicine name column. Found: {list(df.columns)}. "
            "Pass --name_col explicitly."
        )
    if resolved_c1 is None:
        raise KeyError(
            f"Could not find a composition column. Found: {list(df.columns)}. "
            "This script needs real composition data -- pass --c1 explicitly."
        )
    return resolved_name, resolved_c1, resolved_c2


# ----------------------------------------------------------------------
# 1. Normalize a composition string into a canonical salt name
#    "Amoxycillin  (500mg) " -> "amoxicillin"
# ----------------------------------------------------------------------
def clean_salt(s, strip_salt_forms=True):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\(.*?\)", " ", s)                      # drop "(500mg)", "(30mg/5ml)"
    # Any '/' left is alias notation, not a second salt:
    # "Paracetamol/Acetaminophen" -> "paracetamol". Keep the first name.
    s = s.split("/")[0]
    s = re.sub(r"[\d.]+\s*(?:mg|mcg|ml|g|iu|%|w/w|w/v)", " ", s)  # stray dosages
    s = re.sub(r"[^a-z\s]", " ", s)                     # letters only
    s = re.sub(r"\s+", " ", s).strip()
    if strip_salt_forms:
        prev = None
        while prev != s:                                # "sodium phosphate" -> base
            prev = s
            s = _SALT_FORM_RE.sub("", s).strip()
    return SPELL.get(s, s)


def split_composition_field(value):
    """Split one field holding two salts, ignoring '/' inside dosage parens."""
    if not isinstance(value, str) or not value.strip():
        return "", ""
    stripped = re.sub(r"\(.*?\)", " ", value)
    parts = re.split(r"\s*[+|]\s*", stripped.strip(), maxsplit=1)
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]


def composition_key(salt1, salt2, strip_salt_forms=True):
    """A brand's identity = the SET of its salts, order-independent."""
    if (not isinstance(salt2, str) or not salt2.strip()) and isinstance(salt1, str):
        if re.search(r"[+|]", re.sub(r"\(.*?\)", " ", salt1)):
            salt1, salt2 = split_composition_field(salt1)

    salts = [clean_salt(salt1, strip_salt_forms), clean_salt(salt2, strip_salt_forms)]
    salts = sorted({x for x in salts if x})
    return " + ".join(salts)


# Strings pandas would read back as NaN (a brand really is named "None").
NA_SENTINELS = {s.lower() for s in pd._libs.parsers.STR_NA_VALUES} | {""}


def is_writable_text(s):
    return isinstance(s, str) and s.strip().lower() not in NA_SENTINELS


def comp_text(key):
    """Human-readable composition string used as a text anchor."""
    return " + ".join(part.title() for part in key.split(" + "))


# ----------------------------------------------------------------------
# 2. Normalize the brand name we actually embed
# ----------------------------------------------------------------------
def clean_name(n):
    """'Augmentin 625 Duo Tablet' -> 'Augmentin 625 Duo'"""
    n = re.sub(r"\s+", " ", str(n)).strip()
    prev = None
    while prev != n:                       # "Tablet SR", then a second form word
        prev = n
        n = _FORM_RE.sub("", n).strip()
    return n or re.sub(r"\s+", " ", str(n)).strip()


def brand_root(n):
    """'Calpol 500mg' -> 'Calpol'. Bare brand token, before any strength."""
    tokens = n.split()
    root = []
    for tok in tokens:
        if re.search(r"\d", tok):          # first token with a digit = strength
            break
        root.append(tok)
    if not root:
        return ""
    root = " ".join(root).strip(" -,")
    # A single short token like "AB" is too ambiguous to train on.
    if len(root) < 3 or not is_writable_text(root):
        return ""
    return root


# ----------------------------------------------------------------------
# 3. Sampling helpers (never enumerate C(n,2) for huge groups)
# ----------------------------------------------------------------------
def cover_pairs(items, rounds, max_pairs, rng):
    """Pairs that cover EVERY brand, instead of sampling a handful per group.

    Each round shuffles the group and pairs neighbours, so a 5,000-brand
    composition contributes ~2,500 pairs touching all 5,000 names rather than
    20 pairs touching 40. Never materializes C(n,2).
    """
    n = len(items)
    if n < 2:
        return []
    total = n * (n - 1) // 2
    if total <= max_pairs:                      # small group: take everything
        pairs = list(itertools.combinations(items, 2))
        rng.shuffle(pairs)
        return pairs

    seen, out = set(), []
    for _ in range(max(1, rounds)):
        shuffled = list(items)
        rng.shuffle(shuffled)
        for i in range(0, n - 1, 2):
            a, b = shuffled[i], shuffled[i + 1]
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append(key)
        if n % 2:                                # don't strand the odd brand out
            a = shuffled[-1]
            b = shuffled[rng.randrange(n - 1)]
            key = (a, b) if a < b else (b, a)
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


# ----------------------------------------------------------------------
# 4. Build the dataset
# ----------------------------------------------------------------------
def build(csv_path, out_dir, name_col="name",
          c1="short_composition1", c2="short_composition2",
          max_pairs_per_group=20, val_frac=0.15,
          max_triplets_per_group=10, add_comp_anchors=True,
          add_brand_roots=True, strip_salt_forms=True,
          drop_discontinued=False, min_group_size=2,
          keep_truncated=False, pair_rounds=1, root_pairs=1,
          triplet_frac=0.25):

    rng = random.Random(42)

    df = pd.read_csv(csv_path)
    print("Columns found:", list(df.columns))
    print("Rows:", len(df))

    name_col, c1, c2 = resolve_columns(df, name_col, c1, c2)
    print(f"Using columns -> name: {name_col!r}, c1: {c1!r}, c2: {c2!r}")

    if drop_discontinued and "Is_discontinued" in df.columns:
        before = len(df)
        df = df[~df["Is_discontinued"].astype(str).str.lower().isin(["true", "1"])]
        print(f"Dropped discontinued: {before - len(df)}")

    salt2_series = df[c2] if c2 is not None else pd.Series([""] * len(df), index=df.index)
    df = df.assign(
        comp_key=[composition_key(a, b, strip_salt_forms)
                  for a, b in zip(df[c1], salt2_series)],
        clean_name=df[name_col].map(clean_name),
    )

    df = df[(df["comp_key"] != "") & df["clean_name"].map(is_writable_text)]
    print("Rows with usable composition:", len(df))

    # The source only stores TWO salt columns. A title advertising three or more
    # strengths ("Diamerth G 750mg/50mg/250mg") therefore has a truncated
    # composition, which would manufacture false positives. Drop those.
    if not keep_truncated:
        n_strengths = df[name_col].str.count(r"\d+\s*(?:mg|mcg|ml|g|iu)")
        n_salts = df["comp_key"].str.count(r" \+ ") + 1
        truncated = n_strengths > n_salts.clip(lower=2)
        print(f"Dropped {int(truncated.sum())} rows whose title implies more salts "
              f"than the 2 recorded columns")
        df = df[~truncated]

    # ---- one composition per brand name; drop names that disagree ----
    per_name = df.groupby("clean_name")["comp_key"].nunique()
    ambiguous = set(per_name[per_name > 1].index)
    if ambiguous:
        print(f"Dropped {len(ambiguous)} brand names with conflicting compositions")
    df = df[~df["clean_name"].isin(ambiguous)].drop_duplicates("clean_name")
    print("Unique brands:", len(df))

    name_to_key = dict(zip(df["clean_name"], df["comp_key"]))

    # ---- brand roots: bare "Calpol" alongside "Calpol 500mg" ----
    # Only keep a root that maps to exactly ONE composition across the dataset,
    # so "Novamox" (amoxicillin) is not confused with "Novamox CV" (+clav).
    root_variants = defaultdict(set)
    if add_brand_roots:
        root_keys = defaultdict(set)
        for name, key in name_to_key.items():
            root = brand_root(name)
            if root and root != name:
                root_keys[root].add(key)
        for name, key in name_to_key.items():
            root = brand_root(name)
            if root and root != name and len(root_keys[root]) == 1 and root not in name_to_key:
                root_variants[key].add(root)
        print(f"Unambiguous brand roots recovered: {sum(len(v) for v in root_variants.values())}")

    # ---- group brands by identical composition ----
    groups = defaultdict(list)
    for name, key in name_to_key.items():
        groups[key].append(name)
    multi = {k: v for k, v in groups.items() if len(v) >= min_group_size}
    print(f"Compositions with >={min_group_size} brands: {len(multi)}")

    # ---- split by COMPOSITION so val brands are unseen in train ----
    keys = sorted(multi)
    rng.shuffle(keys)
    n_val = int(len(keys) * val_frac)
    val_keys, train_keys = set(keys[:n_val]), set(keys[n_val:])
    print(f"train compositions: {len(train_keys)}   val compositions: {len(val_keys)}")

    def make_pairs(key_set):
        rows = []
        for k in key_set:
            brands = sorted(multi[k])
            rows.extend(cover_pairs(brands, pair_rounds, max_pairs_per_group, rng))
            # bare-brand variants pair with their full titles
            for root in sorted(root_variants.get(k, ())):
                for brand in rng.sample(brands, min(root_pairs, len(brands))):
                    rows.append((root, brand))
            # composition text as an explicit anchor
            if add_comp_anchors:
                anchor = comp_text(k)
                for brand in rng.sample(brands, min(3, len(brands))):
                    rows.append((brand, anchor))
        rng.shuffle(rows)
        return pd.DataFrame(rows, columns=["text_a", "text_b"])

    train_pairs = make_pairs(train_keys)
    val_pairs = make_pairs(val_keys)

    # ---- hard negatives: shares a salt, DIFFERENT full composition ----
    # e.g. "Novamox 500" (amoxicillin) vs "Novamox CV 625" (amoxicillin + clav)
    salt_to_keys = defaultdict(list)
    for k in groups:
        for salt in k.split(" + "):
            salt_to_keys[salt].append(k)

    def make_triplets(key_set):
        rows = []
        allowed = key_set
        for k in key_set:
            brands = multi[k]
            salts = k.split(" + ")
            # candidate compositions that share a salt but are not identical
            neigh = {nk for salt in salts for nk in salt_to_keys[salt]
                     if nk != k and nk in allowed}
            if not neigh:
                neigh = {nk for salt in salts for nk in salt_to_keys[salt] if nk != k}
            if not neigh:
                continue
            neigh = sorted(neigh)
            # scale anchors with group size so large groups aren't under-sampled
            n_trip = min(len(brands),
                         max(max_triplets_per_group, int(len(brands) * triplet_frac)))
            for anchor in rng.sample(brands, n_trip):
                positive = rng.choice([b for b in brands if b != anchor])
                neg_key = rng.choice(neigh)
                negative = rng.choice(groups[neg_key])
                rows.append((anchor, positive, negative))
            # bare brand root as an anchor too
            for root in sorted(root_variants.get(k, ())):
                neg_key = rng.choice(neigh)
                rows.append((root, rng.choice(brands), rng.choice(groups[neg_key])))
        rng.shuffle(rows)
        return pd.DataFrame(rows, columns=["anchor", "positive", "hard_negative"])

    train_triplets = make_triplets(train_keys)
    val_triplets = make_triplets(val_keys)

    # ---- save ----
    os.makedirs(out_dir, exist_ok=True)
    train_pairs.to_csv(f"{out_dir}/train_pairs.csv", index=False)
    val_pairs.to_csv(f"{out_dir}/val_pairs.csv", index=False)
    train_triplets.to_csv(f"{out_dir}/train_triplets.csv", index=False)
    val_triplets.to_csv(f"{out_dir}/val_triplets.csv", index=False)

    print("\n=== OUTPUT ===")
    print(f"train_pairs    : {len(train_pairs):>7}  -> {out_dir}/train_pairs.csv")
    print(f"val_pairs      : {len(val_pairs):>7}  -> {out_dir}/val_pairs.csv")
    print(f"train_triplets : {len(train_triplets):>7}  -> {out_dir}/train_triplets.csv")
    print(f"val_triplets   : {len(val_triplets):>7}  -> {out_dir}/val_triplets.csv")
    print("\nSample positive pairs:")
    print(train_pairs.head(6).to_string(index=False))
    print("\nSample hard-negative triplets:")
    print(train_triplets.head(6).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="./data")
    ap.add_argument("--name_col", default="name")
    ap.add_argument("--c1", default="short_composition1")
    ap.add_argument("--c2", default="short_composition2")
    ap.add_argument("--max_pairs_per_group", type=int, default=20,
                    help="Small groups below this size are fully enumerated")
    ap.add_argument("--pair_rounds", type=int, default=1,
                    help="How many times each brand appears in an in-group pair")
    ap.add_argument("--root_pairs", type=int, default=1,
                    help="Pairs generated per recovered bare brand name")
    ap.add_argument("--max_triplets_per_group", type=int, default=10)
    ap.add_argument("--triplet_frac", type=float, default=0.25,
                    help="Fraction of a group's brands used as triplet anchors")
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--min_group_size", type=int, default=2)
    ap.add_argument("--no_comp_anchors", action="store_true",
                    help="Do not pair brands with their composition text")
    ap.add_argument("--no_brand_roots", action="store_true",
                    help="Do not generate bare brand-name variants")
    ap.add_argument("--keep_salt_forms", action="store_true",
                    help="Treat 'Diclofenac Sodium' and 'Diclofenac Potassium' as different")
    ap.add_argument("--drop_discontinued", action="store_true")
    ap.add_argument("--keep_truncated", action="store_true",
                    help="Keep rows whose title implies more salts than the 2 recorded")
    args = ap.parse_args()

    build(args.csv, args.out, args.name_col, args.c1, args.c2,
          max_pairs_per_group=args.max_pairs_per_group,
          val_frac=args.val_frac,
          max_triplets_per_group=args.max_triplets_per_group,
          add_comp_anchors=not args.no_comp_anchors,
          add_brand_roots=not args.no_brand_roots,
          strip_salt_forms=not args.keep_salt_forms,
          drop_discontinued=args.drop_discontinued,
          min_group_size=args.min_group_size,
          keep_truncated=args.keep_truncated,
          pair_rounds=args.pair_rounds,
          root_pairs=args.root_pairs,
          triplet_frac=args.triplet_frac)
