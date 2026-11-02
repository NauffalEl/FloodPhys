import os, sys, json, math, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rasterio
from rasterio.features import rasterize as rio_rasterize
from shapely.geometry import shape
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.inspection import permutation_importance

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED  = os.path.join(ROOT, "Dataset", "processed")
MODEL_DIR  = os.path.join(ROOT, "Program", "Model")
FIGS_DIR   = os.path.join(ROOT, "Outputs", "figures")
MAPS_DIR   = os.path.join(ROOT, "Outputs", "maps")
TABLES_DIR = os.path.join(ROOT, "Outputs", "tables")
DATASET    = os.path.join(ROOT, "Dataset")
for d in (FIGS_DIR, MAPS_DIR, TABLES_DIR):
    os.makedirs(d, exist_ok=True)

FEATURES = ["elevation","slope","twi","hand","dist_river",
            "drainage_density","rainfall","landcover"]
LABEL = "banjir"
MODEL_NAMES = ["logreg","random_forest","xgboost","lightgbm"]
COLORS = ["#e41a1c","#377eb8","#4daf4a","#ff7f00"]

def load_data():
    df_tr = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    df_te = pd.read_csv(os.path.join(PROCESSED, "features_test.csv"))
    X_tr, y_tr = df_tr[FEATURES].values, df_tr[LABEL].values
    X_te, y_te = df_te[FEATURES].values, df_te[LABEL].values
    return X_tr, y_tr, X_te, y_te

def load_models():
    models = {}
    for name in MODEL_NAMES:
        p = os.path.join(MODEL_DIR, f"{name}.joblib")
        if os.path.exists(p):
            models[name] = joblib.load(p)
    return models

# ── 1. ROC Curves ──
def plot_roc(models, X_te, y_te):
    fig, ax = plt.subplots(figsize=(7, 6))
    for (name, model), color in zip(models.items(), COLORS):
        proba = model.predict_proba(X_te)[:, 1]
        fpr, tpr, _ = roc_curve(y_te, proba)
        roc_auc = auc(fpr, tpr)
        label = f"{name} (AUC={roc_auc:.3f})"
        ax.plot(fpr, tpr, color=color, lw=2, label=label)
    ax.plot([0,1],[0,1],"k--",lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — Flood Susceptibility Prediction, Banyumas Regency", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)
    out = os.path.join(FIGS_DIR, "roc_curve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── 2. Feature Importance (permutation + built-in) ──
def plot_feature_importance(models, X_tr, y_tr, X_te, y_te):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    feat_labels = ["Elevation","Slope","TWI","HAND","Dist. to River",
                   "Drainage Density","Rainfall","Land Cover"]

    for ax, (name, model), color in zip(axes, models.items(), COLORS):
        # try built-in importance first
        importance = None
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = np.abs(model.coef_[0])

        if importance is None:
            # fallback: permutation importance on test
            r = permutation_importance(model, X_te, y_te, n_repeats=10, random_state=42)
            importance = r.importances_mean

        idx = np.argsort(importance)
        ax.barh([feat_labels[i] for i in idx], importance[idx], color=color, alpha=0.8)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Importance", fontsize=10)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Feature Importance — Four Models", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "feature_importance.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── 3. Confusion Matrices ──
def plot_confusion(models, X_te, y_te):
    fig, axes = plt.subplots(1, len(models), figsize=(4*len(models), 4))
    if len(models) == 1:
        axes = [axes]
    for ax, (name, model), color in zip(axes, models.items(), COLORS):
        pred = model.predict(X_te)
        cm = confusion_matrix(y_te, pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["Non-Flood","Flood"], fontsize=9)
        ax.set_yticklabels(["Non-Flood","Flood"], fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(name, fontsize=11, fontweight="bold")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i,j], ha="center", va="center",
                        fontsize=14, color="white" if cm[i,j]>cm.max()/2 else "black")
    fig.suptitle("Confusion Matrix", fontsize=13, fontweight="bold")
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, "confusion_matrices.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")

# ── 4. Peta Kerawanan ──
def make_hazard_map(models):
    """Predict flood probability across entire kabupaten raster → 5-class map."""
    print("Generating hazard map (this may take a moment)...")
    dem_path = os.path.join(PROCESSED, "dem_mosaic.tif")
    raster_paths = {
        "elevation":        dem_path,
        "slope":            os.path.join(PROCESSED, "slope.tif"),
        "twi":              os.path.join(PROCESSED, "twi.tif"),
        "hand":             os.path.join(PROCESSED, "hand.tif"),
        "dist_river":       os.path.join(PROCESSED, "dist_river.tif"),
        "drainage_density": os.path.join(PROCESSED, "drainage_density.tif"),
        "landcover":        os.path.join(DATASET, "worldcover_banyumas.tif"),
    }

    # use best model = XGBoost (by F1)
    best_name = "xgboost" if "xgboost" in models else list(models.keys())[0]
    model = models[best_name]
    print(f"  Using model: {best_name}")

    # load reference raster for shape/transform
    with rasterio.open(dem_path) as ref:
        h, w = ref.height, ref.width
        tf = ref.transform
        crs = ref.crs
        dem_arr = ref.read(1).astype(np.float32)

    # load all raster bands
    bands = {}
    for feat, path in raster_paths.items():
        with rasterio.open(path) as src:
            arr = src.read(1).astype(np.float32)
            # resample to match dem if needed
            if arr.shape != (h, w):
                from rasterio.enums import Resampling as Rs
                from rasterio.warp import reproject
                import rasterio.crs as rcrs
                src_crs = src.crs or rcrs.CRS.from_epsg(4326)
                arr2 = np.zeros((h, w), dtype=np.float32)
                reproject(arr, arr2, src_transform=src.transform,
                          src_crs=src_crs, dst_transform=tf, dst_crs=crs,
                          resampling=Rs.bilinear)
                arr = arr2
            bands[feat] = arr

    # rainfall = constant
    rainfall_val = 3921.6
    valid_mask = ~np.isnan(dem_arr)
    rows_idx, cols_idx = np.where(valid_mask)

    # build feature matrix
    X_map = np.column_stack([
        bands["elevation"][rows_idx, cols_idx],
        bands["slope"][rows_idx, cols_idx],
        bands["twi"][rows_idx, cols_idx],
        bands["hand"][rows_idx, cols_idx],
        bands["dist_river"][rows_idx, cols_idx],
        bands["drainage_density"][rows_idx, cols_idx],
        np.full(len(rows_idx), rainfall_val),
        bands["landcover"][rows_idx, cols_idx],
    ])

    # replace any remaining nan with 0 (edge pixels)
    X_map = np.nan_to_num(X_map, nan=0.0)

    # predict in chunks to avoid OOM
    chunk = 500000
    proba_flat = np.zeros(len(rows_idx), dtype=np.float32)
    for i in range(0, len(rows_idx), chunk):
        proba_flat[i:i+chunk] = model.predict_proba(X_map[i:i+chunk])[:, 1]

    # write probability raster
    proba_arr = np.full((h, w), np.nan, dtype=np.float32)
    proba_arr[rows_idx, cols_idx] = proba_flat

    # 5-class kerawanan (quintiles on valid pixels)
    valid_proba = proba_flat
    breaks = np.percentile(valid_proba, [20, 40, 60, 80])
    class_arr = np.zeros((h, w), dtype=np.uint8)
    for r, c, p in zip(rows_idx, cols_idx, proba_flat):
        if   p <= breaks[0]: class_arr[r, c] = 1
        elif p <= breaks[1]: class_arr[r, c] = 2
        elif p <= breaks[2]: class_arr[r, c] = 3
        elif p <= breaks[3]: class_arr[r, c] = 4
        else:                class_arr[r, c] = 5

    # save class raster
    class_raster_path = os.path.join(MAPS_DIR, "peta_kerawanan_kelas.tif")
    with rasterio.open(class_raster_path, "w", driver="GTiff",
                       height=h, width=w, count=1, dtype="uint8",
                       crs=crs, transform=tf, nodata=0, compress="deflate") as dst:
        dst.write(class_arr, 1)
    print(f"  Saved raster: {class_raster_path}")

    # plot
    colors5 = ["#1a9641","#a6d96a","#ffffbf","#fdae61","#d7191c"]
    labels5 = ["Very Low","Low","Moderate","High","Very High"]
    cmap = ListedColormap(colors5)

    fig, ax = plt.subplots(figsize=(10, 9))
    display = class_arr.astype(float)
    display[class_arr == 0] = np.nan
    im = ax.imshow(display, cmap=cmap, vmin=0.5, vmax=5.5,
                   extent=[tf.c, tf.c + tf.a*w, tf.f + tf.e*h, tf.f])
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.set_title(f"Flood Susceptibility Map — Banyumas Regency\n(Model: {best_name.upper()}, 5 Classes)",
                 fontsize=13, fontweight="bold")

    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors5, labels5)]
    ax.legend(handles=patches, loc="lower right", fontsize=9, title="Susceptibility Class")

    # overlay flood points
    df_all = pd.concat([
        pd.read_csv(os.path.join(PROCESSED, "features.csv")),
        pd.read_csv(os.path.join(PROCESSED, "features_test.csv"))
    ])
    flood_pts = df_all[df_all[LABEL] == 1]
    ax.scatter(flood_pts["lon"], flood_pts["lat"], c="blue", s=30,
               marker="^", zorder=5, edgecolors="white", linewidths=0.5)
    ax.legend(handles=patches + [
        mpatches.Patch(color="blue", label="Flood Inventory Points (BPBD)")
    ], loc="lower right", fontsize=9, title="Legend")

    fig.tight_layout()
    out_png = os.path.join(MAPS_DIR, "peta_kerawanan.png")
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved map: {out_png}")

    return class_arr, tf, crs, h, w, labels5

# ── 5. Luas per kelas ──
def calc_area_per_class(class_arr, tf, labels5):
    lat_rad = math.radians(7.45)
    cellx_km = tf.a * 111.32 * math.cos(lat_rad)
    celly_km = abs(tf.e) * 110.54
    cell_area_km2 = cellx_km * celly_km

    rows = []
    for i, label in enumerate(labels5, start=1):
        n = int((class_arr == i).sum())
        rows.append({"kelas": i, "nama_kelas": label,
                     "jumlah_pixel": n,
                     "luas_km2": round(n * cell_area_km2, 2),
                     "persen": 0.0})
    df = pd.DataFrame(rows)
    total = df["luas_km2"].sum()
    df["persen"] = (df["luas_km2"] / total * 100).round(2)
    out = os.path.join(TABLES_DIR, "luas_kelas.csv")
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    print(df.to_string(index=False))
    return df

# ── main ──
def main():
    print("=== VISUALIZE ===")
    X_tr, y_tr, X_te, y_te = load_data()
    models = load_models()
    if not models:
        print("ERROR: tidak ada model .joblib. Jalankan dulu model.py")
        sys.exit(1)
    print(f"Model loaded: {list(models.keys())}")

    plot_roc(models, X_te, y_te)
    plot_feature_importance(models, X_tr, y_tr, X_te, y_te)
    plot_confusion(models, X_te, y_te)
    class_arr, tf, crs, h, w, labels5 = make_hazard_map(models)
    calc_area_per_class(class_arr, tf, labels5)
    print("=== DONE ===")

if __name__ == "__main__":
    main()
