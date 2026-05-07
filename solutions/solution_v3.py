"""
LING 539 Kaggle Competition 2026 - v3
Task: 3-class text classification
  - Label 0: Amazon product reviews
  - Label 1: Positive IMDb movie reviews
  - Label 2: Negative IMDb movie reviews

Approach:
  - Chinese text → label 0 rule (100% accurate)
  - sentence-transformers (all-MiniLM-L6-v2) embeddings (384-dim)
  - Logistic Regression on embeddings
  - Sentence embeddings capture semantic/sentiment signal that TF-IDF misses
"""

import re
import warnings
import os
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sentence_transformers import SentenceTransformer
import joblib

warnings.filterwarnings("ignore")

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
# Main
# ---------------------------------------------------------------------------

def main():
    total_start = time.time()
    os.makedirs("models", exist_ok=True)
    os.makedirs("submissions", exist_ok=True)

    print("Loading data...")
    train, test = load_data()
    print(f"  Train: {train.shape}, Test: {test.shape}")
    print(f"  Label distribution:\n{train.LABEL.value_counts().sort_index()}")

    # -----------------------------------------------------------------------
    # Step 1: Chinese rule — these are always label 0
    # -----------------------------------------------------------------------
    train["chinese"] = train.TEXT.apply(is_chinese)
    test["chinese"]  = test.TEXT.apply(is_chinese)

    print(f"\n  Chinese rows (train): {train.chinese.sum()}  → rule: label 0")
    print(f"  Chinese rows (test):  {test.chinese.sum()}   → rule: label 0")

    train_ml = train[~train.chinese].reset_index(drop=True)
    test_ml  = test[~test.chinese].reset_index(drop=True)

    print(f"  ML train subset: {len(train_ml)} rows")
    print(f"  ML test  subset: {len(test_ml)} rows")

    # -----------------------------------------------------------------------
    # Step 2: Encode with sentence-transformers
    # -----------------------------------------------------------------------
    print("\nLoading sentence-transformers model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("\nEncoding train texts (~20 min on CPU)...")
    t0 = time.time()
    X_train_emb = model.encode(
        train_ml["TEXT"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  done in {time.time()-t0:.1f}s  shape={X_train_emb.shape}")

    print("\nEncoding test texts...")
    t0 = time.time()
    X_test_emb = model.encode(
        test_ml["TEXT"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  done in {time.time()-t0:.1f}s  shape={X_test_emb.shape}")

    y_all = train_ml["LABEL"].values

    # Save embeddings so we don't re-encode if we want to tune
    np.save("models/train_embeddings.npy", X_train_emb)
    np.save("models/test_embeddings.npy",  X_test_emb)
    print("  Embeddings saved to models/")

    # -----------------------------------------------------------------------
    # Step 3: Train/val split and Logistic Regression
    # -----------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_emb, y_all, test_size=0.1, stratify=y_all, random_state=42
    )

    print("\n=== Logistic Regression on sentence embeddings ===")
    clf = LogisticRegression(
        C=4.0,
        max_iter=1000,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )

    print("  Fitting on train split...")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"  done in {time.time()-t0:.1f}s")

    val_preds = clf.predict(X_val)
    score = accuracy_score(y_val, val_preds)
    print(f"  Val accuracy (non-Chinese subset): {score:.4f}")
    print(classification_report(y_val, val_preds, digits=4))

    # Retrain on all non-Chinese data
    print("  Retraining on full non-Chinese data...")
    t0 = time.time()
    clf.fit(X_train_emb, y_all)
    print(f"  done in {time.time()-t0:.1f}s")
    joblib.dump(clf, "models/logreg_v3.pkl")

    # -----------------------------------------------------------------------
    # Step 4: Assemble final submission
    # -----------------------------------------------------------------------
    preds_ml = clf.predict(X_test_emb)

    submission = pd.DataFrame({"ID": test["ID"], "LABEL": -1})
    submission.loc[test.chinese.values,  "LABEL"] = 0
    submission.loc[~test.chinese.values, "LABEL"] = preds_ml

    assert (submission.LABEL == -1).sum() == 0, "Some rows unassigned!"

    out = "submissions/submission_logreg_v3.csv"
    submission.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

    print(f"\n=== Summary (total: {time.time()-total_start:.1f}s) ===")
    print(f"  Val accuracy (non-Chinese): {score:.4f}")
    n_chinese = train.chinese.sum()
    estimated = (score * len(train_ml) + n_chinese) / len(train)
    print(f"  Estimated overall accuracy: {estimated:.4f}")
    print(f"  Submission: {out}")


if __name__ == "__main__":
    main()
