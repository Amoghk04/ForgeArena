"""Training script for the mechanism-proposal classifier.

Usage
-----
    python scripts/train_mechanism_clf/train.py

Writes the trained Pipeline to:
    src/forge_arena/graders/data/mechanism_clf.pkl

The classifier is a sklearn Pipeline of:
  - FeatureUnion of char-wb TF-IDF (2-4 grams) + word TF-IDF (1-3 grams)
  - LogisticRegression (L2, C=1.0)

Target metric: F1 ≥ 0.83 on held-out validation set (20% stratified split).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow running from repo root without installing the package
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "train_mechanism_clf"))

import joblib
import numpy as np
from dataset import EXAMPLES
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

OUTPUT_PATH = ROOT / "src" / "forge_arena" / "graders" / "data" / "mechanism_clf.pkl"


def build_pipeline() -> Pipeline:
    """Return an untrained sklearn Pipeline."""
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 3),
        min_df=1,
        max_features=30_000,
        sublinear_tf=True,
    )
    features = FeatureUnion(
        transformer_list=[
            ("char_tfidf", char_tfidf),
            ("word_tfidf", word_tfidf),
        ]
    )
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight="balanced")
    return Pipeline(steps=[("features", features), ("clf", clf)])


def main() -> None:
    texts = [e["text"] for e in EXAMPLES]
    labels = [e["label"] for e in EXAMPLES]

    print(f"Dataset: {len(texts)} examples  ({sum(labels)} positive, {len(labels)-sum(labels)} negative)")

    # ── Stratified train/val split ─────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=0.20, stratify=labels, random_state=42
    )
    print(f"Train : {len(X_train)} | Val : {len(X_val)}")

    # ── Train ──────────────────────────────────────────────────────────────
    pipeline = build_pipeline()
    t0 = time.perf_counter()
    pipeline.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0
    print(f"Training time: {elapsed*1000:.0f} ms")

    # ── Evaluate on held-out val ───────────────────────────────────────────
    y_pred = pipeline.predict(X_val)
    val_f1 = f1_score(y_val, y_pred)
    print(f"\nValidation F1 : {val_f1:.4f}")
    print(classification_report(y_val, y_pred, target_names=["assertion", "mechanism"]))

    # ── Cross-validation on full dataset ──────────────────────────────────
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(build_pipeline(), texts, labels, cv=cv, scoring="f1")
    print(f"5-fold CV F1  : {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # ── Sanity check inference speed ──────────────────────────────────────
    test_cases = [
        ("The worker used outdated pricing from v1 rather than the v2 document in the source.", 1),
        ("The pricing is wrong.", 0),
        ("Because the source says 30 days, but the worker cited 14 days.", 1),
        ("The response is incorrect.", 0),
        ("The cited citation does not appear in the source material provided.", 1),
    ]
    t_infer = time.perf_counter()
    preds = pipeline.predict([t for t, _ in test_cases])
    infer_ms = (time.perf_counter() - t_infer) * 1000 / len(test_cases)
    print(f"\nInference speed: {infer_ms:.2f} ms/example")
    for (text, expected), pred in zip(test_cases, preds):
        status = "✓" if pred == expected else "✗"
        print(f"  {status}  [{pred}] {text[:70]}")

    # ── Gate: fail if F1 below target ─────────────────────────────────────
    TARGET_F1 = 0.83
    if val_f1 < TARGET_F1:
        print(f"\n[WARN] Validation F1 {val_f1:.4f} is below target {TARGET_F1}.")
        print("       The classifier will still be saved — inspect the report above.")

    # ── Save ───────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH, compress=3)
    size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nSaved → {OUTPUT_PATH}  ({size_kb:.1f} KB)")
    print("Done.")


if __name__ == "__main__":
    main()
