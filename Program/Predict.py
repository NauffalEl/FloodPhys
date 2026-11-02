import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Utils"))
from features import FEATURES

import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--features", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = joblib.load(args.model)
    df = pd.read_csv(args.features)
    df["prediksi"] = model.predict(df[FEATURES])
    if hasattr(model, "predict_proba"):
        df["prob_banjir"] = model.predict_proba(df[FEATURES])[:, 1]
    df.to_csv(args.out, index=False)
    print(f"OK {len(df)} baris -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
