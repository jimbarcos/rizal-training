"""
evaluate_baseline.py — Baseline Metrics for Helsinki-NLP/opus-mt-tl-en
=======================================================================
Evaluates the raw, un-finetuned base model against the test dataset
to establish a baseline for BLEU, chrF, BERTScore, and COMET.
"""

import os
import json
import torch
import evaluate
import numpy as np
from datasets import load_from_disk
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-tl-en"
DATASET_PATH     = "rizal_tokenized_dataset"   # Must be the same dataset used in training
OUTPUT_DIR       = "rizal-baseline-metrics"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print(f"Loading tokenised dataset from '{DATASET_PATH}' …")
tokenized_datasets = load_from_disk(DATASET_PATH)
test_dataset = tokenized_datasets['test']
print(f"  Test set size: {len(test_dataset):,} samples")

# ─────────────────────────────────────────────
# 2. LOAD BASE MODEL + TOKENIZER (NO LORA)
# ─────────────────────────────────────────────
print(f"Loading raw base model ({MODEL_CHECKPOINT}) …")
tokenizer = MarianTokenizer.from_pretrained(MODEL_CHECKPOINT)
model = MarianMTModel.from_pretrained(MODEL_CHECKPOINT, use_safetensors=True)

model.config.bos_token_id           = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.pad_token_id

# ─────────────────────────────────────────────
# 3. METRICS SETUP
# ─────────────────────────────────────────────
print("Loading evaluation metrics…")
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
bertscore_metric = evaluate.load("bertscore")
comet_metric = evaluate.load("comet")

def compute_metrics(eval_preds):
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    inputs = eval_preds.inputs

    if isinstance(preds, tuple):
        preds = preds[0]

    preds  = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    inputs         = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
    decoded_inputs = tokenizer.batch_decode(inputs, skip_special_tokens=True)

    decoded_preds  = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    decoded_inputs = [i.strip() for i in decoded_inputs]

    bleu = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[l] for l in decoded_labels],
    )
    chrf = chrf_metric.compute(
        predictions=decoded_preds,
        references=decoded_labels,
    )
    bertscore = bertscore_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        lang="en"
    )
    comet = comet_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels, 
        sources=decoded_inputs
    )

    return {
        "bleu": round(bleu["score"], 4),
        "chrf": round(chrf["score"], 4),
        "bertscore": round(np.mean(bertscore["f1"]), 4),
        "comet": round(np.mean(comet["scores"]), 4),
    }

# ─────────────────────────────────────────────
# 4. EVALUATION ARGUMENTS
# ─────────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_eval_batch_size=16,
    
    # Crucial for text generation metrics
    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=128,
    include_inputs_for_metrics=True,
    
    fp16=torch.cuda.is_available(),
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    eval_dataset=test_dataset,
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

# ─────────────────────────────────────────────
# 5. EXECUTE EVALUATION
# ─────────────────────────────────────────────
print("\nStarting baseline evaluation... (This may take a few minutes)")
metrics = trainer.evaluate()

# Clean up the output dictionary (remove internal trainer keys)
clean_metrics = {
    "eval_loss": round(metrics.get("eval_loss", 0), 4),
    "eval_bleu": metrics.get("eval_bleu", 0),
    "eval_chrf": metrics.get("eval_chrf", 0),
    "eval_bertscore": metrics.get("eval_bertscore", 0),
    "eval_comet": metrics.get("eval_comet", 0)
}

print("\n=====================================")
print("         BASELINE METRICS            ")
print("=====================================")
for key, value in clean_metrics.items():
    print(f"  {key:<15}: {value}")
print("=====================================\n")

output_file = os.path.join(OUTPUT_DIR, "baseline_metrics.json")
with open(output_file, "w") as f:
    json.dump(clean_metrics, f, indent=4)
    
print(f"Baseline metrics successfully saved to: {output_file}")