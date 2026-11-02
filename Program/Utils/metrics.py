import os, time
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TABLES_DIR = os.path.join(ROOT, "Outputs", "tables")


def count_params(name, model):
    if name == "logreg":
        coef = model.coef_
        return int(coef.size + model.intercept_.size)
    elif name == "random_forest":
        # total nodes across all trees
        return int(sum(t.tree_.node_count for t in model.estimators_))
    elif name == "xgboost":
        # total leaves × depth proxy
        dump = model.get_booster().get_dump()
        return sum(line.count("leaf") for line in dump)
    elif name == "lightgbm":
        dump = model.booster_.dump_model()
        count = [0]
        def count_leaves(node):
            if "leaf_value" in node or "leaf_index" in node:
                count[0] += 1
                return
            for key in ("left_child", "right_child"):
                ch = node.get(key)
                if isinstance(ch, dict):
                    count_leaves(ch)
        for tree in dump.get("tree_info", []):
            count_leaves(tree.get("tree_structure", {}))
        return count[0]
    return -1


def optimal_threshold(y_true, proba):
    p, r, th = precision_recall_curve(y_true, proba)
    f1s = 2 * p * r / np.clip(p + r, 1e-9, None)
    idx = np.argmax(f1s[:-1])  
    return float(th[idx])


def measure_latency(model, X, n_reps=50):
    _ = model.predict_proba(X[:1])  
    _ = model.predict_proba(X[:10])
    times = []
    for _ in range(n_reps):
        t = time.perf_counter()
        _ = model.predict_proba(X[:1])
        times.append((time.perf_counter() - t) * 1000)
    return float(np.median(times))


def save_metrics_table(results):
    os.makedirs(TABLES_DIR, exist_ok=True)
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values("auc", ascending=False).reset_index(drop=True)
    metrics_path = os.path.join(TABLES_DIR, "metrik_model.csv")
    df_res.to_csv(metrics_path, index=False)
    print(f"\n=== SUMMARY (sorted by AUC) ===")
    print(df_res.to_string(index=False))
    print(f"\nSaved: {metrics_path}")
    return df_res, metrics_path
