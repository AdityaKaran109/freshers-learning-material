"""Evaluate a Whisper baseline or LoRA adapter on a prepared dataset split."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_from_disk
from jiwer import cer, wer
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from text_norm import normalize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/bhojpuri")
    parser.add_argument("--split", default="test")
    parser.add_argument("--model", default="openai/whisper-small")
    parser.add_argument("--adapter", help="Path to a LoRA adapter; omit for baseline")
    parser.add_argument("--language", default="hi")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0, help="0 evaluates all rows")
    parser.add_argument("--dump", help="Optional per-utterance JSONL output path")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    dataset = load_from_disk(args.data)[args.split]
    if args.limit:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa"
    )
    if args.adapter:
        model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()

    model.to(device).eval()
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    references: list[str] = []
    hypotheses: list[str] = []
    rows: list[dict[str, str | float]] = []
    for start in range(0, len(dataset), args.batch):
        batch = dataset[start : start + args.batch]
        features = processor.feature_extractor(
            [audio["array"] for audio in batch["audio"]],
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=dtype)

        with torch.no_grad():
            token_ids = model.generate(features, max_new_tokens=200, num_beams=1)
        predictions = processor.batch_decode(token_ids, skip_special_tokens=True)

        for reference, prediction in zip(batch["text"], predictions):
            normalized_reference = normalize(reference)
            normalized_prediction = normalize(prediction)
            if not normalized_reference:
                continue
            references.append(normalized_reference)
            hypotheses.append(normalized_prediction)
            rows.append(
                {
                    "ref": normalized_reference,
                    "hyp": normalized_prediction,
                    "wer": wer(normalized_reference, normalized_prediction),
                }
            )

        print(f"Scored {min(start + len(predictions), len(dataset))}/{len(dataset)}")

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n": len(references),
        "WER": round(wer(references, hypotheses) * 100, 2),
        "CER": round(cer(references, hypotheses) * 100, 2),
    }
    print(json.dumps(result, indent=2))

    if args.dump:
        dump_path = Path(args.dump)
        with dump_path.open("w", encoding="utf-8") as output_file:
            for row in rows:
                output_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-utterance results written to {dump_path}")


if __name__ == "__main__":
    main()
"""
Step 3: Evaluate baseline Whisper vs your LoRA on the held-out test split.

Reports WER and CER. For Devanagari, CER matters as much as WER -- dialect
models often fix the script/morphology while WER stays noisy because of
compound-word segmentation differences.

Usage:
    # baseline
    python evaluate.py --data data/bhojpuri --split test
    # fine-tuned
    python evaluate.py --data data/bhojpuri --split test --adapter runs/bhojpuri-lv3/adapter
"""

import argparse
import json

import torch
from datasets import load_from_disk
from jiwer import wer, cer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

from text_norm import normalize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bhojpuri")
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--adapter", default=None, help="Omit to score the baseline")
    ap.add_argument("--language", default="hi")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="0 = full split")
    ap.add_argument("--dump", default=None, help="Write per-utterance results to JSONL")
    args = ap.parse_args()

    ds = load_from_disk(args.data)[args.split]
    if args.limit:
        ds = ds.select(range(min(args.limit, len(ds))))
    print(f"Scoring {len(ds)} utterances from '{args.split}'")

    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
        print(f"Loaded adapter: {args.adapter}")
    else:
        print("Baseline (no adapter)")

    model.to("cuda").eval()
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    refs, hyps, rows = [], [], []
    for start in range(0, len(ds), args.batch):
        chunk = ds[start:start + args.batch]
        feats = processor.feature_extractor(
            [a["array"] for a in chunk["audio"]],
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            ids = model.generate(feats, max_new_tokens=200, num_beams=1)
        text = processor.batch_decode(ids, skip_special_tokens=True)

        for ref, hyp in zip(chunk["text"], text):
            r, h = normalize(ref), normalize(hyp)
            if not r:
                continue
            refs.append(r)
            hyps.append(h)
            rows.append({"ref": r, "hyp": h, "wer": wer(r, h)})

        if start % (args.batch * 20) == 0:
            print(f"  {start + len(text)}/{len(ds)}")

    result = {
        "model": args.model,
        "adapter": args.adapter,
        "split": args.split,
        "n": len(refs),
        "WER": round(wer(refs, hyps) * 100, 2),
        "CER": round(cer(refs, hyps) * 100, 2),
    }
    print("\n" + json.dumps(result, indent=2))

    if args.dump:
        with open(args.dump, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Per-utterance results -> {args.dump}")
        print("Sort by wer descending and read the worst 30. That is where the "
              "next improvement comes from.")


if __name__ == "__main__":
    main()
