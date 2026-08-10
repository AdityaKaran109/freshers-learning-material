"""
Step 2: LoRA fine-tune Whisper on a Vaani dialect split.

WHY THIS SCRIPT EXISTS
----------------------
Full fine-tuning of whisper-large-v3 (~1.5B params) is slow and memory-heavy.
LoRA trains small adapter matrices injected into attention and MLP layers while
freezing the base model. You get most of the dialect adaptation benefit at a
fraction of the cost.

CHECKPOINT SELECTION (the important part)
-----------------------------------------
This script selects the best checkpoint on **WER**, not eval_loss.

Cross-entropy loss and WER routinely diverge in ASR: loss can creep upward while
WER keeps falling, because the model becomes less confident but more correct.
Selecting on eval_loss therefore picks the wrong checkpoint on a regular basis.
We generate transcripts during eval and score them with the same normalizer used
in training and final evaluation.

Because of that, you do not need to guess the epoch count. Set a generous budget
(default 6) and let early stopping halt when dev WER stops improving.
`load_best_model_at_end` then restores the actual best point, which for ~24h of
audio usually lands somewhere between 2 and 4 epochs.

DESIGN CHOICES FOR DGX SPARK (ARM64 + unified memory)
-----------------------------------------------------
  * bf16 LoRA, NOT 4-bit QLoRA
      bitsandbytes quantization is unreliable on ARM64; with 128 GB unified
      memory there is no need to compress a 1.5B model.
  * attn_implementation="sdpa"
      FlashAttention is painful to build on ARM/Blackwell; PyTorch SDPA is
      fast enough for this workload.
  * On-the-fly mel extraction in the collator
      Pre-computing log-mel features for 24h of audio would use ~30 GB disk
      for negligible speed gain on NVMe.
  * language="hi" (Hindi token)
      Whisper has no Bhojpuri/Maithili language token. The model learns dialect
      from audio-text pairs via LoRA; the Hindi token is the closest proxy.

Usage:
    # normal run
    python train_lora.py --data data/bhojpuri --out runs/bhojpuri-lv3

    # fast debug loop -- catches every bug in ~20 min
    python train_lora.py --data data/bhojpuri --out runs/debug \\
        --model openai/whisper-small --epochs 1 --batch 8 --workers 0 \\
        --eval-steps 50 --dev-subset 100
"""

import argparse
import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import soundfile as sf
import torch
from datasets import Audio, load_from_disk
from jiwer import cer as jiwer_cer
from jiwer import wer as jiwer_wer
from peft import LoraConfig, get_peft_model
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint

from text_norm import normalize


# --------------------------------------------------------------------------- #
# Audio loading
# --------------------------------------------------------------------------- #
def load_audio_array(audio: Dict) -> np.ndarray:
    """Convert a HuggingFace Audio dict into a 1-D float waveform.

    WHY NOT rely on datasets' default decoder?
    ------------------------------------------
    Recent `datasets` versions decode audio via torchcodec, which requires FFmpeg
    libraries that are often missing. Our saved dataset stores raw WAV bytes, so
    we decode with soundfile instead -- already installed, no FFmpeg dependency.
    """
    if audio.get("array") is not None:
        return audio["array"]
    if audio.get("bytes"):
        array, _ = sf.read(io.BytesIO(audio["bytes"]))
    elif audio.get("path"):
        array, _ = sf.read(audio["path"])
    else:
        raise ValueError("Audio entry has no array, bytes, or path")
    # Downmix stereo defensively; Whisper expects mono.
    if getattr(array, "ndim", 1) > 1:
        array = array.mean(axis=1)
    return array


# --------------------------------------------------------------------------- #
# Batching
# --------------------------------------------------------------------------- #
@dataclass
class Collator:
    """Batch raw dataset rows into Whisper training tensors.

    Seq2SeqTrainer expects input_features (log-mel spectrograms) and labels
    (token IDs). The default collator can't read our audio bytes or handle
    Whisper's decoder-start-token convention.
    """

    processor: Any
    decoder_start_token_id: int
    max_label_len: int = 200
    # Must match the model's dtype. The feature extractor always returns float32;
    # the trainer's autocast hides that during training, but generation at eval
    # time runs outside autocast and dies with
    #   "Input type (float) and bias type (struct c10::BFloat16) should be the same"
    dtype: torch.dtype = torch.bfloat16

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        # 1) Decode waveforms and compute log-mel filterbanks.
        audios = [load_audio_array(f["audio"]) for f in features]
        feats = self.processor.feature_extractor(
            audios, sampling_rate=16_000, return_tensors="pt"
        )
        batch = {"input_features": feats.input_features.to(self.dtype)}

        # 2) Tokenize normalized Devanagari reference text.
        labels = self.processor.tokenizer(
            [f["text"] for f in features],
            max_length=self.max_label_len,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )

        # 3) Mask padding so CrossEntropyLoss ignores it.
        ids = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)

        # 4) Trainer prepends decoder_start_token_id itself; drop any duplicate.
        if (ids[:, 0] == self.decoder_start_token_id).all():
            ids = ids[:, 1:]
        batch["labels"] = ids
        return batch


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def build_compute_metrics(processor, log_path: str = None):
    """Return a compute_metrics fn that decodes generated ids and scores WER/CER.

    Uses the SAME normalizer as prepare_data.py and evaluate.py. If these ever
    diverge, your numbers stop being comparable to your own baseline -- which is
    the most common way people publish accidentally inflated results.
    """
    tok = processor.tokenizer
    pad_id = tok.pad_token_id

    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Some transformers versions return a tuple when generating.
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        pred_ids = np.asarray(pred_ids)
        label_ids = np.asarray(label_ids).copy()

        # -100 is the loss-ignore sentinel; swap back to pad before decoding.
        label_ids[label_ids == -100] = pad_id
        pred_ids = np.where(pred_ids < 0, pad_id, pred_ids)

        refs_raw = tok.batch_decode(label_ids, skip_special_tokens=True)
        hyps_raw = tok.batch_decode(pred_ids, skip_special_tokens=True)

        refs, hyps = [], []
        for r, h in zip(refs_raw, hyps_raw):
            r_n, h_n = normalize(r), normalize(h)
            if r_n:  # skip empty references -- they make WER undefined
                refs.append(r_n)
                hyps.append(h_n)

        if not refs:
            return {"wer": 100.0, "cer": 100.0}

        metrics = {
            "wer": round(100 * jiwer_wer(refs, hyps), 3),
            "cer": round(100 * jiwer_cer(refs, hyps), 3),
        }

        # Dump a few examples so you can eyeball what the model is doing.
        if log_path:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "metrics": metrics,
                    "samples": [{"ref": r, "hyp": h} for r, h in zip(refs[:5], hyps[:5])],
                }, ensure_ascii=False) + "\n")

        return metrics

    return compute_metrics


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/bhojpuri",
                    help="Folder written by prepare_data.py (load_from_disk)")
    ap.add_argument("--out", default="runs/bhojpuri-lv3")
    ap.add_argument("--model", default="openai/whisper-large-v3")
    ap.add_argument("--language", default="hi",
                    help="Whisper language token (no Bhojpuri token exists)")

    # Budget generously; early stopping decides where to actually stop.
    ap.add_argument("--epochs", type=float, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2,
                    help="Effective batch = batch x grad_accum")
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="Drop to 5e-4 if dev WER is flat or worsening early on")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8,
                    help="Use 0 if the first step hangs on your machine")

    # Eval cadence / early stopping
    ap.add_argument("--eval-steps", type=int, default=250,
                    help="Evaluate + checkpoint every N optimizer steps")
    ap.add_argument("--dev-subset", type=int, default=400,
                    help="Utterances used for in-training WER. 0 = full dev split. "
                         "Generation is slow; keep this small.")
    ap.add_argument("--patience", type=int, default=4,
                    help="Stop after N evals with no WER improvement "
                         "(4 x 250 = 1000 steps of no progress)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--resume", default="auto",
                    help="'auto' (default): continue from the newest checkpoint in "
                         "--out if one exists, else start fresh. 'none': always "
                         "start fresh. Or pass an explicit checkpoint path.")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # ---- resume ---------------------------------------------------------- #
    # A checkpoint holds optimizer state, LR-scheduler position, RNG state, the
    # step counter and the early-stopping/best-WER history -- not just weights.
    # Resuming therefore continues the run exactly, rather than restarting a new
    # one from partially-trained weights (which would reset the LR schedule and
    # quietly damage the result).
    resume = None
    if args.resume == "none":
        pass
    elif args.resume == "auto":
        resume = get_last_checkpoint(args.out)  # None if the dir has no checkpoints
    else:
        resume = args.resume
        if not os.path.isdir(resume):
            raise SystemExit(f"--resume path does not exist: {resume}")

    if resume:
        print(f"RESUMING from {resume}")
        print("  (optimizer, LR schedule, RNG and step count are all restored)")
        print("  Pass --resume none to force a fresh start instead.")
    else:
        print("Starting a fresh run (no checkpoint found in --out).")

    # ---- data ----------------------------------------------------------- #
    ds = load_from_disk(args.data)

    # decode=False: keep raw bytes, decode in the collator via soundfile.
    ds = ds.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    train_ds = ds["train"]
    dev_ds = ds["dev"]
    if args.dev_subset and args.dev_subset < len(dev_ds):
        # Shuffle first so the subset isn't one speaker or one recording session.
        dev_ds = dev_ds.shuffle(seed=args.seed).select(range(args.dev_subset))

    steps_per_epoch = max(1, len(train_ds) // (args.batch * args.grad_accum))
    print(f"train: {len(train_ds)}  dev(eval): {len(dev_ds)}  "
          f"test: {len(ds['test'])} (untouched)")
    print(f"~{steps_per_epoch} optimizer steps/epoch  |  "
          f"max {int(steps_per_epoch * args.epochs)} steps  |  "
          f"eval every {args.eval_steps}")

    # ---- model ---------------------------------------------------------- #
    processor = WhisperProcessor.from_pretrained(
        args.model, language=args.language, task="transcribe"
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    )

    # Clear Whisper's default forced decoder ids so the trainer controls inputs.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = args.language
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.use_cache = False  # required with gradient_checkpointing

    lora = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    collator = Collator(processor, model.config.decoder_start_token_id)
    compute_metrics = build_compute_metrics(
        processor, log_path=os.path.join(args.out, "eval_samples.jsonl")
    )

    # ---- training args --------------------------------------------------- #
    targs = Seq2SeqTrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        gradient_checkpointing=True,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        num_train_epochs=args.epochs,
        bf16=True,

        # --- step-based eval so we can see mid-epoch optima ---
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,          # must match eval_steps for best-model tracking
        save_total_limit=3,

        # --- generate real transcripts during eval so WER is meaningful ---
        predict_with_generate=True,
        generation_max_length=200,
        generation_num_beams=1,              # greedy: eval is for ranking, not final numbers

        # --- select on WER, not loss ---
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,

        logging_steps=25,
        dataloader_num_workers=args.workers,
        remove_unused_columns=False,         # MUST keep 'audio' and 'text'
        label_names=["labels"],
        report_to="none",
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,                 # NEVER pass test here
        data_collator=collator,
        processing_class=processor.feature_extractor,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    try:
        trainer.train(resume_from_checkpoint=resume)
    except KeyboardInterrupt:
        last = get_last_checkpoint(args.out)
        print("\n" + "=" * 62)
        print("Interrupted by user.")
        if last:
            print(f"Last saved checkpoint: {last}")
            print("Re-run the EXACT same command to continue from it.")
            print("Do NOT delete the --out folder first, or you lose the progress.")
        else:
            print("No checkpoint was written yet -- the first save happens at step "
                  f"{args.eval_steps}. Nothing to resume from.")
        print("=" * 62)
        raise SystemExit(130)

    # ---- save ------------------------------------------------------------ #
    adapter_dir = os.path.join(args.out, "adapter")
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    print("\n" + "=" * 62)
    print(f"best checkpoint : {trainer.state.best_model_checkpoint}")
    print(f"best dev WER    : {trainer.state.best_metric}")
    print(f"adapter saved   : {adapter_dir}")
    print("=" * 62)

    # Write the dev WER curve so you can see where it flattened.
    hist = [h for h in trainer.state.log_history if "eval_wer" in h]
    with open(os.path.join(args.out, "wer_curve.json"), "w") as f:
        json.dump(hist, f, indent=2)
    for h in hist:
        print(f"  step {h['step']:>6}  wer {h['eval_wer']:.2f}  cer {h.get('eval_cer', float('nan')):.2f}")

    print("\nNext: python evaluate.py --data", args.data, "--adapter", adapter_dir)


if __name__ == "__main__":
    main()
