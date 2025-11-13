# src/hf_train.py
import os
import json
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import evaluate
import torch
from typing import List

# load_label_map function loads the label map from the JSON file.
# We open the JSON file and load the label map.
# We return the label map.
def load_label_map(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# prepare_dataset function prepares the dataset for training.
# We load the label map from the JSON file.
# We create a number of labels from the label map.
# We load the train and validation datasets from the CSV files.
# We convert the train and validation datasets to a Hugging Face Dataset.
# We tokenize the train and validation datasets.
# We return the dataset, number of labels, label map, and tokenizer.
def prepare_dataset(train_csv, val_csv, label_map_path, tokenizer_name="microsoft/deberta-v3-small", max_length=256):
    label_map = load_label_map(label_map_path)
    num_labels = len(label_map)
    # loader expects CSV with columns id,text,labels_bin (comma-separated bits)
    def row_to_example(row):
        labels_bin = [int(x) for x in row["labels_bin"].split(",")]
        return {"id": row["id"], "text": row["text"], "labels": labels_bin}
    train = list(load_dataset("csv", data_files=train_csv)["train"])
    val = list(load_dataset("csv", data_files=val_csv)["train"])
    # convert to HF Dataset
    ds_train = Dataset.from_list([row_to_example(r) for r in train])
    ds_val = Dataset.from_list([row_to_example(r) for r in val])
    ds = DatasetDict({"train": ds_train, "validation": ds_val})
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)

    def tokenize_batch(batch):
        toks = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=max_length)
        toks["labels"] = batch["labels"]
        return toks

    ds = ds.map(tokenize_batch, batched=True, remove_columns=["id","text","labels"])
    ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    return ds, num_labels, label_map, tokenizer

# compute_metrics function computes the metrics for the training.
# We load the F1 metric from the evaluate library.
# We compute the micro and macro F1 scores.
# We return the metrics.
def compute_metrics(pred):
    metric_f1_micro = evaluate.load("f1")
    metric_f1_macro = evaluate.load("f1")
    logits, labels = pred
    probs = torch.sigmoid(torch.tensor(logits))
    y_pred = (probs >= 0.5).int().numpy()
    y_true = labels
    # compute micro & macro F1 (sklearn-style)
    micro = metric_f1_micro.compute(predictions=y_pred.flatten(), references=y_true.flatten(), average="micro")["f1"]
    macro = metric_f1_macro.compute(predictions=y_pred.flatten(), references=y_true.flatten(), average="macro")["f1"]
    return {"f1_micro": micro, "f1_macro": macro}

# train function trains the model.
# We prepare the dataset for training.
# We create the model.
# We set the problem type to multi-label classification.
# We create the training arguments.
# We create the trainer.
# We train the model.
# We save the model.
# We save the tokenizer and label map.
def train(train_csv, val_csv, label_map_path, model_name="microsoft/deberta-v3-small", output_dir="models/mla", epochs=3, batch_size=8):
    ds, num_labels, label_map, tokenizer = prepare_dataset(train_csv, val_csv, label_map_path, tokenizer_name=model_name)
    print("Num labels:", num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # ensure problem type multi-label (BCEWithLogits)
    model.config.problem_type = "multi_label_classification"

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_micro",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(output_dir)
    # save tokenizer + label_map
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print("Model saved to", output_dir)
