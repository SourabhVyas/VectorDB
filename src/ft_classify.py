from pathlib import Path
import json
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from typing import List, Optional

def safe_name(model_name: str) -> str:
    return str(model_name).replace("/", "_").replace(":", "_")

def build_hf_dataset_from_records(records: List[dict], text_field="statement"):
    texts = [r[text_field] for r in records]
    labels = [r.get("numeric_label") for r in records]
    labels = [int(l) for l in labels]
    return Dataset.from_dict({"text": texts, "labels": labels}), [r["id"] for r in records]

def tokenize_batch(batch, tokenizer, max_len=256):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_len)

def fine_tune_base(model_name: str,
                   preproc_json: str,
                   output_dir: Optional[str] = None,
                   epochs: int = 3,
                   per_device_train_batch_size: int = 16,
                   cache_dir: Optional[str] = None,
                   include_unsure: bool = False) -> str:
    preproc_json = Path(preproc_json)
    recs = json.load(open(preproc_json, "r", encoding="utf8"))

    if include_unsure:
        records = [r for r in recs if r.get("numeric_label") in (0,1,2)]
    else:
        records = [r for r in recs if r.get("numeric_label") in (0,1)]

    if not records:
        raise ValueError("No suitable records found for fine-tuning.")

    if output_dir is None:
        base_cache = Path(cache_dir) if cache_dir else Path("./cache")
        output_dir = base_cache / "models" / f"ft_{safe_name(model_name)}"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if (output_dir / "pytorch_model.bin").exists() or any(output_dir.glob("*.bin")):
        print("Found existing checkpoint, skipping:", str(output_dir))
        return str(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    cfg = AutoConfig.from_pretrained(model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=cfg)

    ds, ids = build_hf_dataset_from_records(records)
    ds = ds.map(lambda x: tokenize_batch(x, tokenizer), batched=True)
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=1e-5,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=64,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="roc_auc",
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to="none",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        import numpy as _np
        from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, average_precision_score
        import torch as _torch
        p = _torch.softmax(_torch.tensor(logits), dim=-1)[:, 1].numpy()
        pred = (p >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(labels, pred)),
            "f1": float(f1_score(labels, pred)),
            "roc_auc": float(roc_auc_score(labels, p)),
            "pr_auc": float(average_precision_score(labels, p))
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset=ds,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return str(output_dir)