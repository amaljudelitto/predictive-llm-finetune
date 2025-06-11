
# Fine-Tuning LLaMA 2 with LoRA on AI Instruction Corpus

This project demonstrates fine-tuning the LLaMA-2 7B model on an instruction-following dataset using LoRA and 4-bit quantization.

## Highlights
- Hugging Face `transformers`, `datasets`, `peft`
- Uses AI-focused dataset `mlabonne/guanaco-cleaned`
- Designed for **Google Colab**

## Setup

```bash
pip install transformers datasets peft bitsandbytes accelerate
```

## Training

```bash
python src/train.py
```

## Inference

```bash
python src/infer.py
```

## Output

Model is saved in `./results`
