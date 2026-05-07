"""
LING 539 Kaggle Competition 2026
Task: 3-class text classification
  - Label 0: Amazon product reviews
  - Label 1: Positive IMDb movie reviews
  - Label 2: Negative IMDb movie reviews

Two models:
  1. TF-IDF + Logistic Regression (course requirement)
  2. TF-IDF (word+char n-grams) + LinearSVC (high-performance)
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
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def clean_texts_with_progress(texts, desc="Cleaning text"):
    return [clean_text(t) for t in tqdm(texts, desc=desc, ncols=80)]


def extract_meta_features(texts):
    feats = []
    for t in tqdm(texts, desc="Meta features", ncols=80):
        if not isinstance(t, str):
            t = ""
        has_br   = 1 if re.search(r"<br", t, re.I) else 0
        has_html = 1 if re.search(r"<[a-z]", t, re.I) else 0
        raw_len  = len(t)
        clean    = clean_text(t)
        word_cnt = len(clean.split())
        avg_word = raw_len / max(word_cnt, 1)
        feats.append([has_br, has_html, raw_len, word_cnt, avg_word])
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
    print(f"\n[{split}] Step 1/3: Cleaning text...")
    cleaned = clean_texts_with_progress(raw_texts)

    print(f"[{split}] Step 2/3: Word TF-IDF ({('fit+' if fit else '')}transform)...")
    t0 = time.time()
    X_word = word_tfidf.fit_transform(cleaned) if fit else word_tfidf.transform(cleaned)
    print(f"  done in {time.time()-t0:.1f}s  shape={X_word.shape}")

    print(f"[{split}] Step 3/3: Char TF-IDF ({('fit+' if fit else '')}transform)...")
    t0 = time.time()
    X_char = char_tfidf.fit_transform(cleaned) if fit else char_tfidf.transform(cleaned)
    print(f"  done in {time.time()-t0:.1f}s  shape={X_char.shape}")

    X_meta = extract_meta_features(raw_texts)
    return hstack([X_word, X_char, X_meta])


# ---------------------------------------------------------------------------
# Model 1: TF-IDF + Logistic Regression  (course requirement)
# ---------------------------------------------------------------------------

def run_logistic_regression(X_tr, y_tr, X_val, y_val, X_test):
    print("\n=== Model 1: TF-IDF + Logistic Regression ===")
    clf = LogisticRegression(
        C=4.0, max_iter=500, solver="lbfgs",
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    print("  Fitting on train split...")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"  done in {time.time()-t0:.1f}s")
    score = accuracy_score(y_val, clf.predict(X_val))
    print(f"  Val accuracy: {score:.4f}")

    print("  Retraining on full data...")
    t0 = time.time()
    clf.fit(vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    print(f"  done in {time.time()-t0:.1f}s")

    preds = clf.predict(X_test)
    joblib.dump(clf, "models/logreg.pkl")
    return preds, score


# ---------------------------------------------------------------------------
# Model 2: TF-IDF + LinearSVC  (high-performance)
# ---------------------------------------------------------------------------

def run_linear_svc(X_tr, y_tr, X_val, y_val, X_test):
    print("\n=== Model 2: TF-IDF + LinearSVC ===")
    clf = LinearSVC(C=0.5, max_iter=2000, class_weight="balanced", random_state=42)
    print("  Fitting on train split...")
    t0 = time.time()
    clf.fit(X_tr, y_tr)
    print(f"  done in {time.time()-t0:.1f}s")
    score = accuracy_score(y_val, clf.predict(X_val))
    print(f"  Val accuracy: {score:.4f}")

    print("  Retraining on full data...")
    t0 = time.time()
    clf.fit(vstack([X_tr, X_val]), np.concatenate([y_tr, y_val]))
    print(f"  done in {time.time()-t0:.1f}s")

    preds = clf.predict(X_test)
    joblib.dump(clf, "models/linear_svc.pkl")
    return preds, score


# ---------------------------------------------------------------------------
# Submission helper
# ---------------------------------------------------------------------------

def save_submission(test_ids, preds, filename):
    sub = pd.DataFrame({"ID": test_ids, "LABEL": preds.astype(int)})
    sub.to_csv(filename, index=False)
    print(f"  Saved → {filename}")


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

    word_tfidf = TfidfVectorizer(
        ngram_range=(1, 2), max_features=80_000, sublinear_tf=True,
        min_df=2, strip_accents="unicode", analyzer="word", token_pattern=r"\w{1,}",
    )
    char_tfidf = TfidfVectorizer(
        ngram_range=(3, 5), max_features=50_000, sublinear_tf=True,
        min_df=3, strip_accents="unicode", analyzer="char_wb",
    )

    print("\n--- Building train features ---")
    X_all = build_features(train["TEXT"].tolist(), word_tfidf, char_tfidf, fit=True, split="train")
    print(f"\n--- Building test features ---")
    X_test = build_features(test["TEXT"].tolist(), word_tfidf, char_tfidf, fit=False, split="test")
    y_all = train["LABEL"].values
    print(f"\nFull feature matrix: {X_all.shape}")

    joblib.dump(word_tfidf, "models/word_tfidf.pkl")
    joblib.dump(char_tfidf, "models/char_tfidf.pkl")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_all, test_size=0.1, stratify=y_all, random_state=42
    )

    preds_lr,  score_lr  = run_logistic_regression(X_tr, y_tr, X_val, y_val, X_test)
    preds_svc, score_svc = run_linear_svc(X_tr, y_tr, X_val, y_val, X_test)

    save_submission(test["ID"], preds_lr,  "submissions/submission_logreg.csv")
    save_submission(test["ID"], preds_svc, "submissions/submission_linear_svc.csv")

    print(f"\n=== Summary (total time: {time.time()-total_start:.1f}s) ===")
    print(f"  Logistic Regression val accuracy: {score_lr:.4f}")
    print(f"  LinearSVC           val accuracy: {score_svc:.4f}")
    best = "submissions/submission_logreg.csv" if score_lr > score_svc else "submissions/submission_linear_svc.csv"
    print(f"  Best submission: {best}")


if __name__ == "__main__":
    main()
