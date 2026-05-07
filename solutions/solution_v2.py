"""
LING 539 Kaggle Competition 2026 - v2
Task: 3-class text classification
  - Label 0: Amazon product reviews
  - Label 1: Positive IMDb movie reviews
  - Label 2: Negative IMDb movie reviews

Improvements over v1:
  - Chinese text pre-classified as label 0 (100% accurate rule)
  - Char TF-IDF on RAW text (preserves HTML tag signals)
  - Removed class_weight="balanced"
  - Richer meta features (is_chinese, text length, HTML presence)
  - LogReg only (was better than LinearSVC in v1)
"""

import re
import html
import warnings
import os
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix, vstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Language detection
# ---------------------------------------------------------------------------

def is_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', str(text)))


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    """Strip HTML, decode entities, lowercase — for word TF-IDF only."""
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def extract_meta_features(texts):
    """
    Structural features:
      - is_chinese  : near-perfect label 0 indicator
      - has_br      : <br> tags signal IMDb (labels 1/2)
      - has_html    : any HTML tag
      - raw_len     : raw character length
      - word_count  : number of whitespace tokens
      - avg_word_len: raw_len / word_count
    """
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data():
    train = pd.read_csv("data/train.csv").dropna(subset=["TEXT"])
    test  = pd.read_csv("data/test.csv")
    test["TEXT"] = test["TEXT"].fillna("")
    return train, test


# ---------------------------------------------------------------------------
# Feature building
# ---------------------------------------------------------------------------

def build_features(raw_texts, word_tfidf, char_tfidf, fit=False, split="train"):
    print(f"\n[{split}] Step 1/3: Cleaning text for word TF-IDF...")
    t0 = time.time()
    cleaned = [clean_text(t) for t in tqdm(raw_texts, ncols=80)]
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"[{split}] Step 2/3: Word TF-IDF on cleaned text...")
    t0 = time.time()
    X_word = word_tfidf.fit_transform(cleaned) if fit else word_tfidf.transform(cleaned)
    print(f"  done in {time.time()-t0:.1f}s  shape={X_word.shape}")

    # KEY FIX: char TF-IDF on RAW text — preserves <br>, <b>, etc.
    print(f"[{split}] Step 3/3: Char TF-IDF on RAW text...")
    t0 = time.time()
    X_char = char_tfidf.fit_transform(raw_texts) if fit else char_tfidf.transform(raw_texts)
    print(f"  done in {time.time()-t0:.1f}s  shape={X_char.shape}")

    X_meta = extract_meta_features(raw_texts)
    return hstack([X_word, X_char, X_meta])


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
    # Step 1: Rule — Chinese text is always label 0
    # -----------------------------------------------------------------------
    train["chinese"] = train.TEXT.apply(is_chinese)
    test["chinese"]  = test.TEXT.apply(is_chinese)

    print(f"\n  Chinese rows in train: {train.chinese.sum()} (all label 0)")
    print(f"  Chinese rows in test:  {test.chinese.sum()} (will assign label 0 directly)")

    # Split train into Chinese (rule-handled) and non-Chinese (ML)
    train_ml = train[~train.chinese].copy()
    train_ch = train[train.chinese].copy()

    test_ml  = test[~test.chinese].copy()
    test_ch  = test[test.chinese].copy()

    print(f"\n  ML train subset: {len(train_ml)} rows")
    print(f"  ML test  subset: {len(test_ml)} rows")

    # -----------------------------------------------------------------------
    # Step 2: Build features for non-Chinese subset
    # -----------------------------------------------------------------------
    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=80_000,
        sublinear_tf=True,
        min_df=2,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5),
        max_features=50_000,
        sublinear_tf=True,
        min_df=3,
        analyzer="char_wb",  # no strip_accents — keep < > chars
    )

    print("\n--- Building train features (non-Chinese) ---")
    X_all  = build_features(train_ml["TEXT"].tolist(), word_tfidf, char_tfidf, fit=True,  split="train")
    print("\n--- Building test features (non-Chinese) ---")
    X_test_ml = build_features(test_ml["TEXT"].tolist(), word_tfidf, char_tfidf, fit=False, split="test")
    y_all = train_ml["LABEL"].values

    print(f"\nFeature matrix: {X_all.shape}")

    joblib.dump(word_tfidf, "models/word_tfidf_v2.pkl")
    joblib.dump(char_tfidf, "models/char_tfidf_v2.pkl")

    # -----------------------------------------------------------------------
    # Step 3: Train/val split for evaluation
    # -----------------------------------------------------------------------
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
    )

    # -----------------------------------------------------------------------
    # Step 4: Logistic Regression (course requirement)
    # -----------------------------------------------------------------------
    print("\n=== Logistic Regression (v2) ===")
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

    # Retrain on full non-Chinese data
    print("  Retraining on full non-Chinese data...")
    t0 = time.time()
    clf.fit(X_all, y_all)
    print(f"  done in {time.time()-t0:.1f}s")
    joblib.dump(clf, "models/logreg_v2.pkl")

    # -----------------------------------------------------------------------
    # Step 5: Predict and merge with Chinese rule
    # -----------------------------------------------------------------------
    preds_ml = clf.predict(X_test_ml)

    # Assemble final predictions
    submission = pd.DataFrame({"ID": test["ID"], "LABEL": -1})
    submission.loc[test.chinese.values,  "LABEL"] = 0          # rule: Chinese → 0
    submission.loc[~test.chinese.values, "LABEL"] = preds_ml   # ML predictions

    assert (submission.LABEL == -1).sum() == 0, "Some rows unassigned!"

    out = "submissions/submission_logreg_v2.csv"
    submission.to_csv(out, index=False)
    print(f"\n  Saved → {out}")

    # Overall val accuracy estimate (accounting for Chinese rule)
    n_chinese_train = len(train_ch)
    n_total_train   = len(train)
    weighted_acc    = (score * len(train_ml) + n_chinese_train) / n_total_train
    print(f"\n=== Summary (total: {time.time()-total_start:.1f}s) ===")
    print(f"  Val accuracy (non-Chinese, 10% split): {score:.4f}")
    print(f"  Estimated overall accuracy (incl. Chinese rule): {weighted_acc:.4f}")
    print(f"  Best submission: {out}")


if __name__ == "__main__":
    main()
