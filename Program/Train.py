import os, sys, time
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score,
    f1_score, precision_score, recall_score
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Program", "Utils"))
sys.path.insert(0, os.path.join(ROOT, "Program", "Model"))
from features import FEATURES, LABEL
from model import build_models
from metrics import count_params, optimal_threshold, measure_latency, save_metrics_table

PROCESSED = os.path.join(ROOT, "Dataset", "processed")
MODEL_DIR = os.path.join(ROOT, "Program", "Model")
os.makedirs(MODEL_DIR, exist_ok=True)


def load_data():
    train_path = os.path.join(PROCESSED, "features.csv")
    test_path = os.path.join(PROCESSED, "features_test.csv")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} belum ada. Jalankan dulu: py -3.11 Program/preprocess.py")
        sys.exit(1)
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    X_tr = df_train[FEATURES].values
    y_tr = df_train[LABEL].values
    X_te = df_test[FEATURES].values
    y_te = df_test[LABEL].values
    return X_tr, y_tr, X_te, y_te


def main():
    print("=" * 60)
    print("Centive26 — Model Training & Evaluation")
    print("=" * 60)

    X_tr, y_tr, X_te, y_te = load_data()
    print(f"Train: {X_tr.shape}, Test: {X_te.shape}")
    print(f"Fitur: {FEATURES}")

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"Pos weight (imbalance ratio): {pos_w:.2f}")

    models = build_models(pos_w)
    results = []

    for name, model in models.items():
        print(f"\n--- {name} ---")
        t0 = time.time()
        model.fit(X_tr, y_tr)
        train_time = time.time() - t0
        print(f"  Train time: {train_time:.1f}s")

        out_path = os.path.join(MODEL_DIR, f"{name}.joblib")
        joblib.dump(model, out_path)
        print(f"  Saved: {out_path}")

        proba_tr = model.predict_proba(X_tr)[:, 1]
        proba_te = model.predict_proba(X_te)[:, 1]

        thr_opt = optimal_threshold(y_tr, proba_tr)
        pred_te = (proba_te >= 0.5).astype(int)
        pred_te_opt = (proba_te >= thr_opt).astype(int)

        auc = roc_auc_score(y_te, proba_te)
        acc = accuracy_score(y_te, pred_te)
        f1_macro = f1_score(y_te, pred_te, average="macro", zero_division=0)
        f1_weighted = f1_score(y_te, pred_te, average="weighted", zero_division=0)
        f1_flood = f1_score(y_te, pred_te, pos_label=1, zero_division=0)
        prec = precision_score(y_te, pred_te, pos_label=1, zero_division=0)
        rec = recall_score(y_te, pred_te, pos_label=1, zero_division=0)
        f1_opt = f1_score(y_te, pred_te_opt, pos_label=1, zero_division=0)

        params = count_params(name, model)
        latency_ms = measure_latency(model, X_te)

        print(f"  AUC:         {auc:.4f}")
        print(f"  Accuracy:    {acc:.4f}")
        print(f"  F1-macro:    {f1_macro:.4f}")
        print(f"  F1-weighted: {f1_weighted:.4f}")
        print(f"  F1-banjir:   {f1_flood:.4f} (thr=0.5)  / {f1_opt:.4f} (thr_opt={thr_opt:.3f})")
        print(f"  Precision:   {prec:.4f}")
        print(f"  Recall:      {rec:.4f}")
        print(f"  Params:      {params:,}")
        print(f"  Latency:     {latency_ms:.3f} ms/sample")
        print(classification_report(y_te, pred_te,
              target_names=["non-banjir", "banjir"], zero_division=0))

        results.append({
            "model": name,
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_weighted": round(f1_weighted, 4),
            "f1_banjir_05": round(f1_flood, 4),
            "f1_banjir_opt": round(f1_opt, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "threshold_opt": round(thr_opt, 4),
            "params": params,
            "latency_ms": round(latency_ms, 4),
            "train_time_s": round(train_time, 2),
        })

    df_res, _ = save_metrics_table(results)

    best = df_res.iloc[0]["model"]
    print(f"\nBest model: {best}")
    print("Lanjut: py -3.11 Program/visualize.py")


if __name__ == "__main__":
    main()
