"""
LING 539 Kaggle Competition 2026 - v5
Fine-tune distilbert-base-uncased-finetuned-sst-2-english for 3-class classification.

Model: distilbert-base-uncased-finetuned-sst-2-english
  - Already pre-trained on sentiment (SST-2 movie reviews)
  - 66M parameters, 6 transformer layers, hidden size 768
  - Head replaced with 3-class output (Amazon / IMDb+ / IMDb-)
  - Expected: ~20-25 min/epoch on CPU

Pipeline:
  - Chinese text -> label 0 rule (same as v2/v4)
  - Fine-tune DistilBERT-SST2 on non-Chinese subset
  - 2 epochs, batch size 32, AdamW lr=2e-5
"""

import re
import warnings
import os
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

warnings.filterwarnings("ignore")

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
MAX_LEN    = 256
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 2e-5
SEED       = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def load_data():
    train = pd.read_csv("data/train.csv").dropna(subset=["TEXT"])
    test  = pd.read_csv("data/test.csv")
    test["TEXT"] = test["TEXT"].fillna("")
    return train, test


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ReviewDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ---------------------------------------------------------------------------
# Train / eval helpers
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0
    for batch in tqdm(loader, desc="  train", ncols=80, leave=False):
        batch   = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss    = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        preds = outputs.logits.argmax(dim=-1)
        total_loss    += loss.item() * len(preds)
        total_correct += (preds == batch["labels"]).sum().item()
        total         += len(preds)
    return total_loss / total, total_correct / total


def eval_epoch(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval ", ncols=80, leave=False):
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return np.array(all_preds)


def predict(model, loader, device):
    model.eval()
    all_preds = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  predict", ncols=80, leave=False):
            batch  = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
    return np.array(all_preds)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load data + Chinese rule
    # -----------------------------------------------------------------------
    print("\nLoading data...")
    train, test = load_data()

    train["chinese"] = train.TEXT.apply(is_chinese)
    test["chinese"]  = test.TEXT.apply(is_chinese)

    train_ml = train[~train.chinese].reset_index(drop=True)
    test_ml  = test[~test.chinese].reset_index(drop=True)

    print(f"  ML train: {len(train_ml)}  |  ML test: {len(test_ml)}")
    print(f"  Chinese rule covers {test.chinese.sum()} test rows -> label 0")

    # -----------------------------------------------------------------------
    # Tokenizer + model
    # NOTE: num_labels=3 replaces the SST-2 binary head with a 3-class head.
    #       The transformer body retains its sentiment-aware weights.
    # -----------------------------------------------------------------------
    print(f"\nLoading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=3, ignore_mismatched_sizes=True
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # -----------------------------------------------------------------------
    # Train/val split
    # -----------------------------------------------------------------------
    texts  = train_ml["TEXT"].tolist()
    labels = train_ml["LABEL"].values

    tr_texts, val_texts, tr_labels, val_labels = train_test_split(
        texts, labels, test_size=0.1, stratify=labels, random_state=SEED
    )

    train_ds = ReviewDataset(tr_texts,  tr_labels,  tokenizer, MAX_LEN)
    val_ds   = ReviewDataset(val_texts, val_labels, tokenizer, MAX_LEN)
    test_ds  = ReviewDataset(test_ml["TEXT"].tolist(), None, tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # -----------------------------------------------------------------------
    # Optimizer + scheduler
    # -----------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps,
    )

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    print(f"\nFine-tuning {MODEL_NAME} for {EPOCHS} epochs...")
    print(f"  Train batches/epoch: {len(train_loader)}  |  batch size: {BATCH_SIZE}")
    best_val_acc    = 0
    best_model_path = "models/distilbert_sst2_best.pt"

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"  train loss: {train_loss:.4f}  train acc: {train_acc:.4f}  ({time.time()-t0:.0f}s)")

        val_preds = eval_epoch(model, val_loader, device)
        val_acc   = accuracy_score(val_labels, val_preds)
        print(f"  val   acc:  {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  saved best model (val acc: {best_val_acc:.4f})")

    # -----------------------------------------------------------------------
    # Load best model + final report
    # -----------------------------------------------------------------------
    print(f"\nLoading best model (val acc: {best_val_acc:.4f})...")
    model.load_state_dict(torch.load(best_model_path, map_location=device))

    print("Final val classification report:")
    val_preds = eval_epoch(model, val_loader, device)
    print(classification_report(val_labels, val_preds,
          target_names=["Amazon", "IMDb+", "IMDb-"], digits=4))

    # Retrain on full data for final test predictions
    print("Retraining on full non-Chinese data for final predictions...")
    full_ds     = ReviewDataset(texts, labels, tokenizer, MAX_LEN)
    full_loader = DataLoader(full_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    optimizer2 = AdamW(model.parameters(), lr=LR * 0.5, weight_decay=0.01)
    scheduler2 = get_linear_schedule_with_warmup(
        optimizer2,
        num_warmup_steps=len(full_loader) // 10,
        num_training_steps=len(full_loader) * 2,
    )
    for ep in range(2):
        t0 = time.time()
        loss, acc = train_epoch(model, full_loader, optimizer2, scheduler2, device)
        print(f"  retrain epoch {ep+1}: loss={loss:.4f} acc={acc:.4f} ({time.time()-t0:.0f}s)")
    torch.save(model.state_dict(), "models/distilbert_sst2_final.pt")

    # -----------------------------------------------------------------------
    # Generate submission
    # -----------------------------------------------------------------------
    print("\nPredicting on test set...")
    preds_ml = predict(model, test_loader, device)

    submission = pd.DataFrame({"ID": test["ID"], "LABEL": -1})
    submission.loc[test.chinese.values,  "LABEL"] = 0
    submission.loc[~test.chinese.values, "LABEL"] = preds_ml

    assert (submission.LABEL == -1).sum() == 0

    out = "submissions/submission_distilbert_sst2.csv"
    submission.to_csv(out, index=False)
    print(f"\n  Saved -> {out}")

    print(f"\n=== Summary (total: {time.time()-total_start:.0f}s) ===")
    print(f"  Best val accuracy: {best_val_acc:.4f}")
    print(f"  Submission: {out}")


if __name__ == "__main__":
    main()
