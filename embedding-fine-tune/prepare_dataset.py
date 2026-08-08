r"""
prepare_dataset.py
------------------
Turn the A-Z Medicine Dataset of India into training data for embedding
fine-tuning.

Goal: two brands with the SAME active-ingredient composition should embed
close together, and two brands whose NAMES look alike but whose compositions
differ should not.

    Mox 500mg      = Amoxicillin                 \  same molecule -> CLOSE
    Novamox 500    = Amoxicillin                 /

    Ibugesic 400   = Ibuprofen                   \  near-identical names,
    Ibugesic Plus  = Ibuprofen + Paracetamol     /  different -> FAR APART

The second case is the hard one, and it drives three design decisions:

1. SPLIT BY BRAND, NOT BY COMPOSITION.
   Holding out whole compositions asks the model to recognise a molecule it
   has never seen named even once, which a brand name alone cannot support.
   Instead every composition stays in training and unseen BRANDS are held
   out -- the real task, "a new brand launches for a known salt".
   A small set of whole compositions is still held out separately as a
   deliberately hard secondary metric.

2. MINE LOOK-ALIKE NEGATIVES FROM BRAND ROOTS.
   ~2.8k brand roots span more than one composition (Ibugesic / Ibugesic
   Plus, Mox / Mox CV). These are exactly the pairs models get wrong, so
   they become explicit hard negatives instead of being discarded.

3. GROUND EVERY BRAND TO ITS COMPOSITION TEXT.
   Pairing "Mox 500mg" with "Amoxicillin" gives the model a target the
   biomedical base model already understands, rather than only pushing
   arbitrary brand strings toward each other.

Produces in --out:
    train_pairs.csv            (text_a, text_b)
    train_triplets.csv         (anchor, positive, hard_negative)
    val_pairs.csv              unseen brands, seen compositions
    val_triplets.csv           unseen brands, seen compositions   <- primary
    val_lookalike_triplets.csv only root-collision negatives       <- the hard metric
    unseen_comp_triplets.csv   compositions never trained on       <- secondary

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

# Strings pandas would read back as NaN (a brand really is named "None").
NA_SENTINELS = {s.lower() for s in pd._libs.parsers.STR_NA_VALUES} | {""}


def is_writable_text(s):
    return isinstance(s, str) and s.strip().lower() not in NA_SENTINELS


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


def comp_text(key):
    """Human-readable composition string used as a text anchor."""
    return " + ".join(part.title() for part in key.split(" + "))


# ----------------------------------------------------------------------
# 2. Normalize the brand name we actually embed
# ----------------------------------------------------------------------
def clean_name(n):
    """'Augmentin 625 Duo Tablet' -> 'Augmentin 625 Duo'"""
    original = re.sub(r"\s+", " ", str(n)).strip()
    n = original
    prev = None
    while prev != n:                       # "Tablet SR", then a second form word
        prev = n
        n = _FORM_RE.sub("", n).strip()
    return n or original


def brand_root(n):
    """'Calpol 500mg' -> 'Calpol'. Bare brand token, before any strength."""
    root = []
    for tok in n.split():
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
          max_pairs_per_group=20, pair_rounds=1, root_pairs=1,
          max_triplets_per_group=10, triplet_frac=0.25,
          root_collision_anchors=6, prefix_len=5, prefix_collision_anchors=2,
          val_frac=0.12, unseen_comp_frac=0.05,
          add_comp_anchors=True, add_brand_roots=True,
          strip_salt_forms=True, drop_discontinued=False,
          min_group_size=2, keep_truncated=False):

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
    groups = defaultdict(list)
    for name, key in name_to_key.items():
        groups[key].append(name)
    multi = {k: sorted(v) for k, v in groups.items() if len(v) >= min_group_size}
    print(f"Compositions with >={min_group_size} brands: {len(multi)}")

    # ---- root index: which compositions does each brand root cover? ----
    root_index = defaultdict(lambda: defaultdict(list))   # root -> key -> [names]
    for name, key in name_to_key.items():
        root = brand_root(name)
        if root and root != name:
            root_index[root][key].append(name)
    collide = {r: ks for r, ks in root_index.items() if len(ks) > 1}
    print(f"Brand roots spanning >1 composition (look-alike negatives): {len(collide)}")

    # Root matching is exact, so it misses "Mox 500mg" vs "Moxikind-CV 625".
    # A shared character prefix catches those: same opening letters, different
    # composition. Coincidental collisions are fine here -- the pair still looks
    # alike to a tokenizer, which is precisely what must be told apart.
    prefix_index = defaultdict(lambda: defaultdict(list))
    for name, key in name_to_key.items():
        flat = re.sub(r"[^a-z0-9]", "", name.lower())
        if len(flat) >= prefix_len:
            prefix_index[flat[:prefix_len]][key].append(name)
    prefix_collide = {p: ks for p, ks in prefix_index.items() if len(ks) > 1}
    print(f"Name prefixes ({prefix_len} chars) spanning >1 composition: {len(prefix_collide)}")

    # ------------------------------------------------------------------
    # SPLIT 1: a few whole compositions held out entirely (hard metric)
    # ------------------------------------------------------------------
    splittable = [k for k, v in multi.items() if len(v) >= 4]
    rng.shuffle(splittable)
    n_unseen = int(len(multi) * unseen_comp_frac)
    unseen_keys = set(splittable[:n_unseen])
    seen_keys = [k for k in multi if k not in unseen_keys]
    print(f"Held-out compositions (never trained): {len(unseen_keys)}")

    # ------------------------------------------------------------------
    # SPLIT 2: within every remaining composition, hold out BRANDS
    # ------------------------------------------------------------------
    train_of, val_of = {}, {}
    for k in seen_keys:
        brands = list(multi[k])
        rng.shuffle(brands)
        # need >=2 brands on each side to form a pair
        n_val = int(len(brands) * val_frac)
        if len(brands) < 6:
            n_val = 0
        n_val = max(0, min(n_val, len(brands) - 2))
        if n_val == 1:
            n_val = 0
        val_of[k] = sorted(brands[:n_val])
        train_of[k] = sorted(brands[n_val:])
    n_train_brands = sum(len(v) for v in train_of.values())
    n_val_brands = sum(len(v) for v in val_of.values())
    print(f"Brands -> train: {n_train_brands}   val (unseen brand, seen composition): "
          f"{n_val_brands}")

    # Brand roots are derived from TRAIN brands only, and only when the root maps
    # to exactly one composition -- "Novamox" must not be merged with "Novamox CV".
    root_alias = defaultdict(set)
    if add_brand_roots:
        train_names = {n for v in train_of.values() for n in v}
        for root, keys in root_index.items():
            if len(keys) != 1 or root in name_to_key:
                continue
            key = next(iter(keys))
            if key in train_of and any(n in train_names for n in keys[key]):
                root_alias[key].add(root)
        print(f"Unambiguous brand roots usable as training aliases: "
              f"{sum(len(v) for v in root_alias.values())}")

    # ------------------------------------------------------------------
    # Pair / triplet construction, run once per split
    # ------------------------------------------------------------------
    def make_pairs(split_of, with_roots):
        rows = []
        for k, brands in split_of.items():
            if len(brands) < 2:
                continue
            rows.extend(cover_pairs(brands, pair_rounds, max_pairs_per_group, rng))
            if with_roots:
                for root in sorted(root_alias.get(k, ())):
                    for brand in rng.sample(brands, min(root_pairs, len(brands))):
                        rows.append((root, brand))
            # ground EVERY brand to its composition text
            if add_comp_anchors:
                anchor = comp_text(k)
                for brand in brands:
                    rows.append((brand, anchor))
        rng.shuffle(rows)
        return pd.DataFrame(rows, columns=["text_a", "text_b"])

    # compositions that share at least one salt -- the "looks related" pool
    salt_to_keys = defaultdict(list)
    for k in multi:
        for salt in k.split(" + "):
            salt_to_keys[salt].append(k)

    def make_triplets(split_of, with_roots, lookalike_only=False, neg_of=None):
        """Three kinds of hard negative.

        A) shares a salt, different full composition  (Mox vs Mox CV)
        B) shares a brand ROOT, different composition (Ibugesic vs Ibugesic Plus)
        C) composition text as anchor, grounded against a salt-sharing rival

        anchor and positive always come from split_of, so they are unseen at
        evaluation time. neg_of supplies the negatives and defaults to the same
        split; evaluation sets widen it to every brand, since a negative the
        model trained on still makes a valid "is the positive closer?" test and
        root collisions are too rare to survive a 12% slice.
        """
        rows, counts = [], defaultdict(int)
        available = {k: set(v) for k, v in split_of.items()}
        neg_of = split_of if neg_of is None else neg_of
        neg_available = {k: set(v) for k, v in neg_of.items()}

        # ---- B) look-alike negatives from colliding brand roots ----
        for root, keys in collide.items():
            for k in keys:
                if not available.get(k):
                    continue
                mine = [n for n in keys[k] if n in available[k]]
                others = [n for ok, names in keys.items() if ok != k
                          for n in names if n in neg_available.get(ok, ())]
                pool = [b for b in split_of[k] if b not in mine]
                if not mine or not others or not pool:
                    continue
                for anchor in rng.sample(mine, min(root_collision_anchors, len(mine))):
                    rows.append((anchor, rng.choice(pool), rng.choice(others)))
                    counts["root_collision"] += 1

        # ---- D) look-alike negatives from shared name prefixes ----
        for prefix, keys in prefix_collide.items():
            for k in keys:
                if not available.get(k):
                    continue
                mine = [n for n in keys[k] if n in available[k]]
                others = [n for ok, names in keys.items() if ok != k
                          for n in names if n in neg_available.get(ok, ())]
                pool = [b for b in split_of[k] if b not in mine]
                if not mine or not others or not pool:
                    continue
                for anchor in rng.sample(mine, min(prefix_collision_anchors, len(mine))):
                    rows.append((anchor, rng.choice(pool), rng.choice(others)))
                    counts["prefix_collision"] += 1
        if lookalike_only:
            rng.shuffle(rows)
            return pd.DataFrame(rows, columns=["anchor", "positive", "hard_negative"]), counts

        # ---- A) salt-overlap negatives ----
        for k, brands in split_of.items():
            if len(brands) < 2:
                continue
            neigh = [nk for salt in k.split(" + ") for nk in salt_to_keys[salt]
                     if nk != k and neg_available.get(nk)]
            if not neigh:
                continue
            neigh = sorted(set(neigh))
            n_trip = min(len(brands),
                         max(max_triplets_per_group, int(len(brands) * triplet_frac)))
            for anchor in rng.sample(brands, n_trip):
                positive = rng.choice([b for b in brands if b != anchor])
                negative = rng.choice(sorted(neg_available[rng.choice(neigh)]))
                rows.append((anchor, positive, negative))
                counts["salt_overlap"] += 1

            # ---- C) composition text anchored against a rival composition ----
            if add_comp_anchors:
                neg_key = rng.choice(neigh)
                rows.append((comp_text(k), rng.choice(brands),
                             rng.choice(sorted(neg_available[neg_key]))))
                counts["comp_anchor"] += 1

            if with_roots:
                for root in sorted(root_alias.get(k, ())):
                    neg_key = rng.choice(neigh)
                    rows.append((root, rng.choice(brands),
                                 rng.choice(sorted(neg_available[neg_key]))))
                    counts["root_alias"] += 1

        rng.shuffle(rows)
        return pd.DataFrame(rows, columns=["anchor", "positive", "hard_negative"]), counts

    unseen_of = {k: multi[k] for k in unseen_keys}

    train_pairs = make_pairs(train_of, with_roots=True)
    val_pairs = make_pairs(val_of, with_roots=False)
    train_triplets, train_counts = make_triplets(train_of, with_roots=True)
    # Evaluation sets keep anchor+positive unseen but draw negatives from the
    # whole catalogue, so root collisions are not decimated by the 12% slice.
    val_triplets, val_counts = make_triplets(val_of, with_roots=False, neg_of=multi)
    val_lookalike, look_counts = make_triplets(val_of, with_roots=False,
                                               lookalike_only=True, neg_of=multi)
    unseen_triplets, unseen_counts = make_triplets(unseen_of, with_roots=False, neg_of=multi)

    # ---- save ----
    os.makedirs(out_dir, exist_ok=True)
    outputs = {
        "train_pairs": train_pairs,
        "val_pairs": val_pairs,
        "train_triplets": train_triplets,
        "val_triplets": val_triplets,
        "val_lookalike_triplets": val_lookalike,
        "unseen_comp_triplets": unseen_triplets,
    }
    for stem, frame in outputs.items():
        frame.to_csv(f"{out_dir}/{stem}.csv", index=False)

    print("\n=== OUTPUT ===")
    for stem, frame in outputs.items():
        print(f"{stem:24}: {len(frame):>7}  -> {out_dir}/{stem}.csv")
    print(f"\ntrain triplet mix: {dict(train_counts)}")
    print(f"val   triplet mix: {dict(val_counts)}")
    print(f"val look-alike only: {dict(look_counts)}")

    print("\nSample look-alike triplets (anchor / same composition / LOOKS alike but differs):")
    print(val_lookalike.head(8).to_string(index=False))


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
    ap.add_argument("--val_frac", type=float, default=0.12,
                    help="Fraction of BRANDS held out inside each composition")
    ap.add_argument("--unseen_comp_frac", type=float, default=0.05,
                    help="Fraction of whole compositions held out entirely")
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
          pair_rounds=args.pair_rounds,
          root_pairs=args.root_pairs,
          max_triplets_per_group=args.max_triplets_per_group,
          triplet_frac=args.triplet_frac,
          val_frac=args.val_frac,
          unseen_comp_frac=args.unseen_comp_frac,
          add_comp_anchors=not args.no_comp_anchors,
          add_brand_roots=not args.no_brand_roots,
          strip_salt_forms=not args.keep_salt_forms,
          drop_discontinued=args.drop_discontinued,
          min_group_size=args.min_group_size,
          keep_truncated=args.keep_truncated)
