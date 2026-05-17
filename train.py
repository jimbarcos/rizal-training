"""
train.py — Rizal TL→EN LoRA fine-tune, version 3
===================================================
"""

import os
import json
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, TaskType

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MODEL_CHECKPOINT = "Helsinki-NLP/opus-mt-tl-en"
DATASET_PATH     = "rizal_tokenized_dataset"   # built by preprocess3.py
OUTPUT_DIR       = "rizal-mt-lora-v3"
ADAPTER_SAVE_DIR = "rizal-lora-adapters-v3-final"
METRICS_LOG_DIR  = "rizal-metrics-graphs"      # NEW: Folder for metric graphs

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("Loading tokenised dataset …")
tokenized_datasets = load_from_disk(DATASET_PATH)
print(f"  Train: {len(tokenized_datasets['train']):,}  "
      f"  Eval:  {len(tokenized_datasets['test']):,}")

# ─────────────────────────────────────────────
# 2. LOAD MODEL + TOKENIZER
# ─────────────────────────────────────────────
print("Loading base model and tokeniser …")
tokenizer = MarianTokenizer.from_pretrained(MODEL_CHECKPOINT)
model     = MarianMTModel.from_pretrained(MODEL_CHECKPOINT, use_safetensors=True)

model.config.bos_token_id           = tokenizer.pad_token_id
model.config.decoder_start_token_id = tokenizer.pad_token_id

model.gradient_checkpointing_enable()

# ─────────────────────────────────────────────
# 3. LoRA  (v3: higher rank + explicit cross-attn)
# ─────────────────────────────────────────────
print("Injecting LoRA adapters (v3: r=48, alpha=64) …")
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=48,
    lora_alpha=64,
    lora_dropout=0.15,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "out_proj",
        "fc1", "fc2",
    ],
    bias="none",
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ─────────────────────────────────────────────
# 4. METRICS  (sacrebleu + chrF + BERTScore + COMET)
# ─────────────────────────────────────────────
bleu_metric = evaluate.load("sacrebleu")
chrf_metric = evaluate.load("chrf")
bertscore_metric = evaluate.load("bertscore")
comet_metric = evaluate.load("comet")

def compute_metrics(eval_preds):
    # Because we set include_inputs_for_metrics=True, eval_preds now contains the inputs
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    inputs = eval_preds.inputs

    if isinstance(preds, tuple):
        preds = preds[0]

    preds  = np.clip(preds, 0, tokenizer.vocab_size - 1)

    decoded_preds  = tokenizer.batch_decode(preds,  skip_special_tokens=True)
    labels         = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Decode inputs for COMET
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
# 5. CUSTOM CALLBACK FOR LOGGING & PLOTTING GRAPHS
# ─────────────────────────────────────────────
class MetricsPlottingCallback(TrainerCallback):
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {
            "epoch": [],
            "eval_loss": [],
            "eval_bleu": [],
            "eval_chrf": [],
            "eval_bertscore": [],
            "eval_comet": []
        }

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return

        # Extract current epoch and metrics
        epoch = metrics.get("epoch", state.epoch)
        self.history["epoch"].append(epoch)
        self.history["eval_loss"].append(metrics.get("eval_loss", 0))
        self.history["eval_bleu"].append(metrics.get("eval_bleu", 0))
        self.history["eval_chrf"].append(metrics.get("eval_chrf", 0))
        self.history["eval_bertscore"].append(metrics.get("eval_bertscore", 0))
        self.history["eval_comet"].append(metrics.get("eval_comet", 0))

        # Save raw data to JSON
        with open(os.path.join(self.output_dir, "metrics_history.json"), "w") as f:
            json.dump(self.history, f, indent=4)

        # Generate plots
        self._plot_metric("eval_loss", "Evaluation Loss", "Loss", "red")
        self._plot_metric("eval_bleu", "BLEU Score", "BLEU", "blue")
        self._plot_metric("eval_chrf", "chrF Score", "chrF", "green")
        self._plot_metric("eval_bertscore", "BERTScore (F1)", "Score", "purple")
        self._plot_metric("eval_comet", "COMET Score", "Score", "orange")

    def _plot_metric(self, metric_key, title, ylabel, color):
        epochs = self.history["epoch"]
        values = self.history[metric_key]
        
        plt.figure(figsize=(8, 5))
        plt.plot(epochs, values, marker='o', linestyle='-', color=color, label=title)
        plt.title(f"{title} per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel(ylabel)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.tight_layout()
        
        filepath = os.path.join(self.output_dir, f"{metric_key}_plot.png")
        plt.savefig(filepath, dpi=300)
        plt.close()

# ─────────────────────────────────────────────
# 6. TRAINING ARGUMENTS
# ─────────────────────────────────────────────
args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,

    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=4,
    load_best_model_at_end=True,
    metric_for_best_model="comet",
    greater_is_better=True,
    
    include_inputs_for_metrics=True, # Required for COMET to see the Tagalog text

    learning_rate=3e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=2,
    weight_decay=0.01,
    max_grad_norm=1.0,
    num_train_epochs=50,
    lr_scheduler_type="cosine",
    warmup_steps=150,
    label_smoothing_factor=0.1,

    predict_with_generate=True,
    generation_num_beams=4,
    generation_max_length=128,

    fp16=torch.cuda.is_available(),
    gradient_checkpointing=True,

    logging_strategy="steps",
    logging_steps=50,
    report_to="none",
)

# ─────────────────────────────────────────────
# 7. TRAINER
# ─────────────────────────────────────────────
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

trainer = Seq2SeqTrainer(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    data_collator=data_collator,
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=5,
            early_stopping_threshold=0.0,
        ),
        MetricsPlottingCallback(output_dir=METRICS_LOG_DIR)  # NEW: Added custom callback here
    ],
)

# ─────────────────────────────────────────────
# 8. TRAIN
# ─────────────────────────────────────────────
print("Starting PEFT training…")
trainer.train()

print("Training complete!  Saving LoRA adapters …")
trainer.save_model(ADAPTER_SAVE_DIR)
tokenizer.save_pretrained(ADAPTER_SAVE_DIR)
print(f"Adapters and tokeniser saved to: {ADAPTER_SAVE_DIR}/")
print(f"Metrics graphs and history saved to: {METRICS_LOG_DIR}/")

# ─────────────────────────────────────────────
# 9. QUICK SANITY-CHECK INFERENCE
# ─────────────────────────────────────────────
print("\n--- Sanity-check inference ---")
model.eval()
device  = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

samples = [
    # Modern Tagalog
    "Ang buhay ng isang tao ay hindi sukat na mahalaga sa kanya sa lahat ng bagay.",
    "Sa aking pagkabata ay natutuhan ko na ang katotohanan.",
    "Ang kalayaan ay hindi kailanman ibibigay ng malupit na may kapangyarihan.",
    "Ang kabataang hindi marunong lumingon sa pinanggalingan ay hindi makakarating sa paroroonan.",
    
    # 19th-Century / Old Tagalog Equivalents (Rizal-era Orthography & Vocabulary)
    "Ang buhay nang isang tauo ay hindi sucat na mahalaga sa caniya sa lahat nang bagay.",
    "Sa aquing camusmusan ay naalaman co na ang catotohanan.",
    "Ang calayaan ay di cailan man ipagcacaloob nang malupit na may capangyarihan.",
    "Ang cabataang hindi marunong lumingon sa pinangalingan ay hindi macacarating sa paroroonan.",
    
    #Classical/Poetic Tagalog Equivalents
    "Ang hininga ng isáng nilaláng ay dî sukat ituring na higit sa lahat ng nanga-sa ibabaw ng lupà.",
    "Nang ako'y musmos pa lamang ay nabatid ko na ang liwanag ng katwiran.",
    "Ang kasarinlan ay kailanma'y dî isusuko ng palalong naghahari-harian.",
    "Siyang dî marunong mag-alaala sa kaniyang tinalikdan ay dî sasapit sa kaniyang mithiin."
]

inputs = tokenizer(
    samples,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128,
).to(device)

with torch.no_grad():
    translated = model.generate(**inputs, num_beams=4, max_new_tokens=128)

for src, tgt in zip(samples, tokenizer.batch_decode(translated, skip_special_tokens=True)):
    print(f"  SRC : {src}")
    print(f"  TGT : {tgt}\n")