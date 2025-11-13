# src/data_prep.py
"""
Convert a CSV/DB with columns (id, text, labels) into Hugging Face Dataset ready for training.
Produces:
 - train.csv / val.csv (if doing split)
 - label_map.json
 - optionally dataset_arrow (HF dataset saved)
"""
import csv
import json
import os
from collections import defaultdict
from typing import List
from sklearn.model_selection import train_test_split

# load_csv function loads the CSV file into a list of dictionaries.
# We open the CSV file and read the rows into a list of dictionaries.
# We return the list of dictionaries.
def load_csv(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

# build_label_map function builds the label map from the rows.
# We create a set of labels from the rows.
# We sort the labels and create a label map.
# We return the label map.
def build_label_map(rows):
    labels = set()
    for r in rows:
        raw = r.get("labels","")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        labels.update(parts)
    label_list = sorted(labels)
    label_map = {lab: i for i, lab in enumerate(label_list)}
    return label_map

# rows_to_multihot function converts the rows to a multi-hot encoded format.
# We create a list of texts, labels, and ids.
# We iterate over the rows and convert the labels to a multi-hot encoded format.
# We return the ids, texts, and labels.
def rows_to_multihot(rows, label_map):
    X = []
    Y = []
    ids = []
    for r in rows:
        text = r.get("text") or ""
        raw = r.get("labels","")
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        vec = [0] * len(label_map)
        for p in parts:
            if p in label_map:
                vec[label_map[p]] = 1
        X.append(text)
        Y.append(vec)
        ids.append(r.get("id"))
    return ids, X, Y

# split_and_save function splits the data into train and validation sets and saves the data to a CSV file.
# We load the CSV file into a list of dictionaries.
# We build the label map from the rows.
# We convert the rows to a multi-hot encoded format.
# We split the data into train and validation sets.
# We save the data to a CSV file.
# We return the paths to the train, validation, and label map files.
def split_and_save(input_csv, out_dir="data", test_size=0.1, random_state=42):
    os.makedirs(out_dir, exist_ok=True)
    rows = load_csv(input_csv)
    label_map = build_label_map(rows)
    ids, X, Y = rows_to_multihot(rows, label_map)

    # split
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(X, Y, ids, test_size=test_size, random_state=random_state)

    # save new CSVs (labels as pipe-separated binary string)
    def write_csv(xs, ys, ids, path):
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["id","text","labels_bin"])
            for i, x in enumerate(xs):
                writer.writerow([ids[i], x, ",".join(str(int(v)) for v in ys[i])])

    write_csv(X_train, y_train, ids_train, os.path.join(out_dir, "train.csv"))
    write_csv(X_val, y_val, ids_val, os.path.join(out_dir, "val.csv"))
    with open(os.path.join(out_dir, "label_map.json"), "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"Saved train/val and label_map ({len(label_map)} classes) to {out_dir}")
    return os.path.join(out_dir, "train.csv"), os.path.join(out_dir, "val.csv"), os.path.join(out_dir, "label_map.json")
