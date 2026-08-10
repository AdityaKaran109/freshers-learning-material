"""Download and prepare the official Vaani Bhojpuri splits for Whisper."""

import argparse
import os

from datasets import Audio, DatasetDict, load_dataset

from text_norm import is_usable, normalize


DATASET = "ARTPARK-IISc/Vaani-transcription-part"
KEEP_COLUMNS = {"audio", "text", "district", "gender", "referenceImage"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare official Vaani splits without repartitioning them."
    )
    parser.add_argument("--language", default="Bhojpuri")
    parser.add_argument("--out", default="data/bhojpuri")
    args = parser.parse_args()

    dataset = load_dataset(DATASET, args.language, token=os.environ.get("HF_TOKEN"))
    required_splits = {"train", "validation", "test"}
    missing_splits = required_splits - set(dataset)
    if missing_splits:
        raise SystemExit(
            f"Expected official train/validation/test splits; missing: "
            f"{', '.join(sorted(missing_splits))}. Do not re-split the dataset."
        )

    print("Official splits:", {name: len(split) for name, split in dataset.items()})

    splits = DatasetDict(
        {
            "train": dataset["train"],
            "dev": dataset["validation"],
            "test": dataset["test"],
        }
    ).cast_column("audio", Audio(sampling_rate=16_000))

    splits = splits.map(
        lambda batch: {"text": [normalize(text) for text in batch["transcript"]]},
        batched=True,
        batch_size=512,
        desc="Normalizing transcripts",
    )
    splits = splits.filter(
        lambda batch: [is_usable(text) for text in batch["text"]],
        batched=True,
        batch_size=512,
        desc="Dropping unusable transcripts",
    )
    splits = splits.remove_columns(
        [
            column
            for column in splits["train"].column_names
            if column not in KEEP_COLUMNS
        ]
    )

    for name, split in splits.items():
        print(f"{name:>5}: {len(split)}")

    os.makedirs(args.out, exist_ok=True)
    splits.save_to_disk(args.out)
    print(f"Saved prepared data to {args.out}")


if __name__ == "__main__":
    main()
"""
Step 1: Prepare the Vaani Bhojpuri subset for Whisper fine-tuning.

THE ONE RULE: use the dataset's OFFICIAL train/validation/test splits.
Do not concatenate them and re-split.

Vaani has no speaker ID column (schema: audio, language, gender, state,
district, transcript, referenceImage). ARTPARK had the speaker metadata when
they built the splits; we don't. A random re-split puts the same voice in train
and test, the model memorises voices instead of the dialect, and WER drops
20-30 points for no real reason.

Usage:
    huggingface-cli login          # dataset is gated; accept terms on the Hub
    python prepare_data.py --language Bhojpuri --out data/bhojpuri
"""

import argparse
import os

from datasets import Audio, DatasetDict, load_dataset

from text_norm import is_usable, normalize

DATASET = "ARTPARK-IISc/Vaani-transcription-part"
KEEP = {"audio", "text", "district", "gender", "referenceImage"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="Bhojpuri",
                    help="Vaani config: Bhojpuri, Maithili, Magadhi, Hindi, ...")
    ap.add_argument("--out", default="data/bhojpuri")
    args = ap.parse_args()

    raw = load_dataset(DATASET, args.language, token=os.environ.get("HF_TOKEN"))
    print("official splits:", {k: len(v) for k, v in raw.items()})
    print("columns:", raw["train"].column_names)

    if not {"train", "test"} <= set(raw.keys()):
        raise SystemExit("Expected official train/test splits; got "
                         f"{list(raw.keys())}. Do not re-split manually.")

    splits = DatasetDict({
        "train": raw["train"],
        "dev": raw["validation"] if "validation" in raw else raw["train"],
        "test": raw["test"],
    })

    splits = splits.cast_column("audio", Audio(sampling_rate=16_000))
    splits = splits.map(
        lambda b: {"text": [normalize(t) for t in b["transcript"]]},
        batched=True, batch_size=512, desc="normalizing")
    splits = splits.filter(
        lambda b: [is_usable(t) for t in b["text"]],
        batched=True, batch_size=512, desc="dropping junk")

    splits = splits.remove_columns(
        [c for c in splits["train"].column_names if c not in KEEP])

    # Leakage diagnostics -- read these before trusting any score.
    dup = set(splits["train"]["text"]) & set(splits["test"]["text"])
    print(f"\nidentical transcripts in train AND test: {len(dup)} "
          f"of {len(splits['test'])}")
    if "referenceImage" in splits["train"].column_names:
        tr = set(splits["train"]["referenceImage"])
        te = set(splits["test"]["referenceImage"])
        print(f"referenceImage overlap train/test: {len(tr & te)}")
        print("  (shared prompt images = shared topics, not speaker leakage,")
        print("   but mention it in the model card)")

    print()
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v)}")

    os.makedirs(args.out, exist_ok=True)
    splits.save_to_disk(args.out)
    print(f"\nSaved to {args.out}. Do not touch the test split until done.")


if __name__ == "__main__":
    main()
