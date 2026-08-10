# Fine-tune Whisper for Bhojpuri ASR

Reproducible LoRA fine-tuning of `openai/whisper-small` on the Bhojpuri
configuration of [Vaani](https://huggingface.co/datasets/ARTPARK-IISc/Vaani-transcription-part).

The repository contains source code only. It deliberately excludes Vaani data,
model weights, checkpoints, virtual environments, API tokens, and generated
predictions.

## What is included

- `prepare_data.py` downloads, cleans, and preserves Vaani's official splits.
- `text_norm.py` provides the one normalizer used throughout the pipeline.
- `train_lora.py` trains a LoRA adapter and selects checkpoints on development
  WER rather than loss.
- `evaluate.py` calculates WER/CER for the baseline or trained adapter.
- `check_leakage.py` audits evaluation outputs for detectable text leakage.

## Setup

Python 3.12 and a CUDA-capable GPU are recommended. For an RTX 50-series GPU,
install a CUDA 12.8 PyTorch wheel before installing the remaining packages:

```powershell
python -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
```

Vaani is a gated CC-BY-4.0 dataset. Accept its terms on Hugging Face, then
authenticate with `hf auth login` before preparing data.

## Run the pipeline

```powershell
# 1. Download, normalize, and retain the official train/dev/test partitions.
python prepare_data.py --language Bhojpuri --out data/bhojpuri

# 2. Establish a baseline on the untouched test set.
python evaluate.py --data data/bhojpuri --split test `
    --model openai/whisper-small --dump baseline_small.jsonl

# 3. Train the LoRA adapter. Re-run the same command to resume a stopped run.
python train_lora.py --data data/bhojpuri --out runs/bhojpuri-small `
    --model openai/whisper-small --lr 1e-3 --epochs 6 --batch 16

# 4. Evaluate the adapter against the same test set.
python evaluate.py --data data/bhojpuri --split test `
    --model openai/whisper-small `
    --adapter runs/bhojpuri-small/adapter --dump tuned_small.jsonl

# 5. Audit the published score for detectable transcript overlap.
python check_leakage.py --data data/bhojpuri --dump tuned_small.jsonl
```

## Reproducibility notes

- Never concatenate and re-split Vaani. Its official partitions are the only
  available protection against speaker leakage.
- WER and CER use the shared `text_norm.py` normalizer. Scores obtained with
  another normalizer are not directly comparable.
- The default Hindi (`hi`) language token is intentional: Whisper has no
  Bhojpuri token.
- The test split is not used during training or checkpoint selection.

## Attribution and license

The Vaani dataset is CC-BY-4.0. This project redistributes no Vaani audio or
transcripts; users must accept the dataset's terms before downloading it.
