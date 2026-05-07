"""
LING 539 Kaggle Competition 2026 - v4
Task: 3-class text classification
  - Label 0: Amazon product reviews
  - Label 1: Positive IMDb movie reviews
  - Label 2: Negative IMDb movie reviews

Approach: Combined features
  - TF-IDF (word + char on raw text) from v2  → domain/structural signal
  - Sentence embeddings (all-MiniLM-L6-v2) from v3 → semantic/sentiment signal
  - Logistic Regression on concatenated features
  - Chinese rule: Chinese text → label 0 (no ML needed)

Reuses saved artifacts from v2 and v3 — no re-encoding needed.
"""

import re
import html
import warnings
import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Helpers (same as v2)
# ---------------------------------------------------------------------------

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_meta_features(texts):
    feats = []
    for t in tqdm(texts, desc="Meta features", ncols=80):
        if not isinstance(t, str):
            t = ""
        chinese  = 1 if re.search(r'[\u4e00-\u9fff]', t) else 0
        has_br   = 1 if re.search(r"<br", t, re.I) else 0
        has_html = 1 if re.search(r"<[a-z]", t, re.I) else 0
        raw_len  = len(t)
        word_cnt = len(t.split())
        avg_word = raw_len / max(word_cnt, 1)
        feats.append([chinese, has_br, has_html, raw_len, word_cnt, avg_word])
    return csr_matrix(np.array(feats, dtype=np.float32))


def build_tfidf_features(raw_texts, word_tfidf, char_tfidf, fit=False, split=""):
    cleaned = [clean_text(t) for t in tqdm(raw_texts, desc=f"[{split}] clean", ncols=80)]
    X_word = word_tfidf.fit_transform(cleaned) if fit else word_tfidf.transform(cleaned)
    X_char = char_tfidf.fit_transform(raw_texts) if fit else char_tfidf.transform(raw_texts)
    X_meta = extract_meta_features(raw_texts)
    return hstack([X_word, X_char, X_meta])


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

    # -----------------------------------------------------------------------
    # Step 1: Chinese rule
    # -----------------------------------------------------------------------
    train["chinese"] = train.TEXT.apply(is_chinese)
    test["chinese"]  = test.TEXT.apply(is_chinese)

    train_ml = train[~train.chinese].reset_index(drop=True)
    test_ml  = test[~test.chinese].reset_index(drop=True)
    y_all    = train_ml["LABEL"].values

    print(f"  ML train subset: {len(train_ml)} | ML test subset: {len(test_ml)}")

    # -----------------------------------------------------------------------
    # Step 2: TF-IDF features (load saved v2 vectorizers)
    # -----------------------------------------------------------------------
    print("\nLoading saved v2 TF-IDF vectorizers...")
    word_tfidf = joblib.load("models/word_tfidf_v2.pkl")
    char_tfidf = joblib.load("models/char_tfidf_v2.pkl")

    print("Building TF-IDF features (transform only, no refit)...")
    t0 = time.time()
    X_tfidf_train = build_tfidf_features(train_ml["TEXT"].tolist(), word_tfidf, char_tfidf, fit=False, split="train")
    X_tfidf_test  = build_tfidf_features(test_ml["TEXT"].tolist(),  word_tfidf, char_tfidf, fit=False, split="test")
    print(f"  done in {time.time()-t0:.1f}s  shape={X_tfidf_train.shape}")

    # -----------------------------------------------------------------------
    # Step 3: Sentence embeddings (load saved v3 embeddings)
    # -----------------------------------------------------------------------
    print("\nLoading saved v3 sentence embeddings...")
    X_emb_train = np.load("models/train_embeddings.npy")
    X_emb_test  = np.load("models/test_embeddings.npy")
    print(f"  train embeddings: {X_emb_train.shape}")
    print(f"  test  embeddings: {X_emb_test.shape}")

    # -----------------------------------------------------------------------
    # Step 4: Concatenate TF-IDF + embeddings
    # -----------------------------------------------------------------------
    print("\nConcatenating TF-IDF + sentence embeddings...")
    X_train_combined = hstack([X_tfidf_train, csr_matrix(X_emb_train)])
    X_test_combined  = hstack([X_tfidf_test,  csr_matrix(X_emb_test)])
    print(f"  Combined feature matrix: {X_train_combined.shape}")

    # -----------------------------------------------------------------------
    # Step 5: Train/val split + Logistic Regression
    # -----------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_combined, y_all, test_size=0.1, stratify=y_all, random_state=42
    )

    print("\n=== Logistic Regression (TF-IDF + sentence embeddings) ===")
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
    print(f"  Val accuracy (non-Chinese): {score:.4f}")
    print(classification_report(y_val, val_preds, digits=4))

    print("  Retraining on full non-Chinese data...")
    t0 = time.time()
    clf.fit(X_train_combined, y_all)
    print(f"  done in {time.time()-t0:.1f}s")
    joblib.dump(clf, "models/logreg_v4.pkl")

    # -----------------------------------------------------------------------
    # Step 6: Assemble submission
    # -----------------------------------------------------------------------
    preds_ml = clf.predict(X_test_combined)

    submission = pd.DataFrame({"ID": test["ID"], "LABEL": -1})
    submission.loc[test.chinese.values,  "LABEL"] = 0
    submission.loc[~test.chinese.values, "LABEL"] = preds_ml

    assert (submission.LABEL == -1).sum() == 0

    out = "submissions/submission_logreg_v4.csv"
    submission.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

    estimated = (score * len(train_ml) + train.chinese.sum()) / len(train)
    print(f"\n=== Summary (total: {time.time()-total_start:.1f}s) ===")
    print(f"  v2 (TF-IDF only)          val acc: 0.8970")
    print(f"  v3 (embeddings only)       val acc: 0.8798")
    print(f"  v4 (TF-IDF + embeddings)   val acc: {score:.4f}  ← this run")
    print(f"  Estimated overall accuracy: {estimated:.4f}")
    print(f"  Submission: {out}")


if __name__ == "__main__":
    main()
