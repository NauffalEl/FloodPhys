import os, sys, glob, json, math, time
import numpy as np

if not hasattr(np, "in1d"):
    np.in1d = np.isin
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask as rasterio_mask
from rasterio.transform import from_bounds, rowcol
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely.geometry import shape, Point, box
from shapely.ops import unary_union
from scipy.ndimage import distance_transform_edt, uniform_filter
from scipy.spatial import cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(ROOT, "Dataset")
PROCESSED = os.path.join(DATASET, "processed")
os.makedirs(PROCESSED, exist_ok=True)

# ── helpers ──
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def load_boundary():
    gj = json.load(open(os.path.join(DATASET, "banyumas_boundary.geojson"), encoding="utf-8"))
    feats = gj["features"] if gj.get("type") == "FeatureCollection" else [gj]
    geom = shape(feats[0]["geometry"])
    return geom

def write_raster(path, data, transform, crs="EPSG:4326", nodata=np.nan):
    with rasterio.open(path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1],
                       count=1, dtype=data.dtype, crs=crs, transform=transform, nodata=nodata,
                       compress="deflate") as dst:
        dst.write(data, 1)

# ── 1. Mosaic + Clip DEMNAS ──
def mosaic_clip():
    out_path = os.path.join(PROCESSED, "dem_mosaic.tif")
    if os.path.exists(out_path):
        log("dem_mosaic.tif exists, skip")
        return out_path
    log("Mosaic 9 DEMNAS tiles...")
    tiles = sorted(glob.glob(os.path.join(DATASET, "DEMNAS_*.tif")))
    srcs = [rasterio.open(t) for t in tiles]
    mosaic_arr, mosaic_tf = merge(srcs, nodata=np.nan)
    for s in srcs:
        s.close()
    mosaic_arr = mosaic_arr[0]  # single band
    log(f"  mosaic shape {mosaic_arr.shape}, bounds transform")

    # clip to boundary
    log("Clip to kabupaten boundary...")
    boundary = load_boundary()
    # write temp mosaic to clip
    tmp = os.path.join(PROCESSED, "_tmp_mosaic.tif")
    write_raster(tmp, mosaic_arr, mosaic_tf)
    with rasterio.open(tmp) as src:
        clipped, clip_tf = rasterio_mask(src, [boundary], crop=True, nodata=np.nan)
    clipped = clipped[0]
    write_raster(out_path, clipped, clip_tf)
    os.remove(tmp)
    log(f"  clipped shape {clipped.shape}, saved {out_path}")
    return out_path

# ── 2. Slope (degrees) ──
def calc_slope(dem_path):
    out_path = os.path.join(PROCESSED, "slope.tif")
    if os.path.exists(out_path):
        log("slope.tif exists, skip")
        return out_path
    log("Calculate slope...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        tf = src.transform
    # pixel size in meters (approximate at lat -7.4)
    lat_rad = math.radians(7.45)
    cellx = tf.a * 111320 * math.cos(lat_rad)  # degrees to meters
    celly = abs(tf.e) * 110540
    dy, dx = np.gradient(dem, celly, cellx)
    slope = np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))
    slope[np.isnan(dem)] = np.nan
    write_raster(out_path, slope, tf)
    log(f"  slope range {np.nanmin(slope):.2f}–{np.nanmax(slope):.2f} deg")
    return out_path

# ── 3. TWI (Topographic Wetness Index) ──
def calc_twi(dem_path, slope_path):
    out_path = os.path.join(PROCESSED, "twi.tif")
    if os.path.exists(out_path):
        log("twi.tif exists, skip")
        return out_path
    log("Calculate TWI...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        tf = src.transform
    with rasterio.open(slope_path) as src:
        slope = src.read(1).astype(np.float64)

    # simplified flow accumulation via multiple flow direction (Quinn 1991)
    lat_rad = math.radians(7.45)
    cellx = tf.a * 111320 * math.cos(lat_rad)
    celly = abs(tf.e) * 110540
    cell_area = cellx * celly

    rows, cols = dem.shape
   
    from pysheds.grid import Grid

    # write dem to temp for pysheds
    tmp_dem = os.path.join(PROCESSED, "_tmp_twi_dem.tif")
    write_raster(tmp_dem, dem.astype(np.float32), tf)

    grid = Grid.from_raster(tmp_dem)
    dem_grid = grid.read_raster(tmp_dem)
    # fill pits
    pit_filled = grid.fill_pits(dem_grid)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)
    acc_arr = np.array(acc).astype(np.float64)

    # specific catchment area
    slope_rad = np.radians(slope)
    slope_rad[slope_rad < 0.001] = 0.001  # avoid log(0)
    sca = (acc_arr + 1) * cell_area / cellx  # m

    twi = np.log(sca / np.tan(slope_rad))
    twi[np.isnan(dem)] = np.nan
    twi = np.clip(twi, -5, 30).astype(np.float32)
    write_raster(out_path, twi, tf)
    os.remove(tmp_dem)
    log(f"  TWI range {np.nanmin(twi):.2f}–{np.nanmax(twi):.2f}")
    return out_path

# ── 4. HAND (Height Above Nearest Drainage) ──
def calc_hand(dem_path):
    out_path = os.path.join(PROCESSED, "hand.tif")
    if os.path.exists(out_path):
        log("hand.tif exists, skip")
        return out_path
    log("Calculate HAND (Height Above Nearest Drainage)...")
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float64)
        tf = src.transform

    tmp_dem = os.path.join(PROCESSED, "_tmp_hand_dem.tif")
    write_raster(tmp_dem, dem.astype(np.float32), tf)

    from pysheds.grid import Grid
    grid = Grid.from_raster(tmp_dem)
    dem_grid = grid.read_raster(tmp_dem)
    pit_filled = grid.fill_pits(dem_grid)
    flooded = grid.fill_depressions(pit_filled)
    inflated = grid.resolve_flats(flooded)
    fdir = grid.flowdir(inflated)
    acc = grid.accumulation(fdir)
    acc_arr = np.array(acc).astype(np.float64)

    # drainage threshold: top 2% accumulation = stream cells
    valid_acc = acc_arr[~np.isnan(dem)]
    threshold = np.percentile(valid_acc, 98)
    stream_mask = acc_arr >= threshold

    # for each cell, find nearest stream cell and get its elevation
    stream_coords = np.argwhere(stream_mask)
    non_stream_coords = np.argwhere(~stream_mask & ~np.isnan(dem))

    hand = np.full_like(dem, np.nan)
    hand[stream_mask] = 0.0

    if len(stream_coords) > 0 and len(non_stream_coords) > 0:
        tree = cKDTree(stream_coords)
        dists, idxs = tree.query(non_stream_coords)
        nearest_stream = stream_coords[idxs]
        for i, (r, c) in enumerate(non_stream_coords):
            sr, sc = nearest_stream[i]
            hand[r, c] = max(0.0, dem[r, c] - dem[sr, sc])

    hand = hand.astype(np.float32)
    write_raster(out_path, hand, tf)
    os.remove(tmp_dem)
    log(f"  HAND range {np.nanmin(hand):.2f}–{np.nanmax(hand):.2f} m")
    return out_path

# ── 5. Distance to river ──
def calc_dist_river(dem_path):
    out_path = os.path.join(PROCESSED, "dist_river.tif")
    if os.path.exists(out_path):
        log("dist_river.tif exists, skip")
        return out_path
    log("Calculate distance to nearest river...")
    rivers_gj = json.load(open(os.path.join(DATASET, "rivers_banyumas.geojson"), encoding="utf-8"))
    river_lines = [shape(f["geometry"]) for f in rivers_gj["features"]]

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        tf = src.transform
        h, w = dem.shape

    # sample points along rivers every ~100m (~0.001 deg)
    river_pts = []
    for line in river_lines:
        length = line.length
        n = max(1, int(length / 0.001))
        for i in range(n + 1):
            p = line.interpolate(i / n, normalized=True)
            river_pts.append((p.y, p.x))
    river_pts = np.array(river_pts)
    log(f"  {len(river_pts)} river sample points")

    # build KDTree in approximate meters
    lat_rad = math.radians(7.45)
    mx = 111320 * math.cos(lat_rad)
    my = 110540
    river_m = river_pts * [my, mx]
    tree = cKDTree(river_m)

    # for each pixel center, query distance
    dist_arr = np.full((h, w), np.nan, dtype=np.float32)
    for r in range(h):
        y = tf.f + tf.e * (r + 0.5)
        xs = tf.c + tf.a * (np.arange(w) + 0.5)
        coords_m = np.column_stack([np.full(w, y * my), xs * mx])
        valid = ~np.isnan(dem[r])
        if valid.any():
            d, _ = tree.query(coords_m[valid])
            dist_arr[r, valid] = d.astype(np.float32)

    write_raster(out_path, dist_arr, tf)
    log(f"  dist_river range {np.nanmin(dist_arr):.0f}–{np.nanmax(dist_arr):.0f} m")
    return out_path

# ── 6. Drainage density ──
def calc_drainage_density(dem_path):
    out_path = os.path.join(PROCESSED, "drainage_density.tif")
    if os.path.exists(out_path):
        log("drainage_density.tif exists, skip")
        return out_path
    log("Calculate drainage density...")
    rivers_gj = json.load(open(os.path.join(DATASET, "rivers_banyumas.geojson"), encoding="utf-8"))
    river_lines = [shape(f["geometry"]) for f in rivers_gj["features"]]

    with rasterio.open(dem_path) as src:
        dem = src.read(1)
        tf = src.transform
        h, w = dem.shape

    lat_rad = math.radians(7.45)
    cellx_deg = tf.a
    celly_deg = abs(tf.e)
    cellx_km = cellx_deg * 111.32 * math.cos(lat_rad)
    celly_km = celly_deg * 110.54


    river_raster = np.zeros((h, w), dtype=np.float32)
    seg_length_km = math.sqrt(cellx_km**2 + celly_km**2)
    for line in river_lines:
        coords = list(line.coords)
        for i in range(len(coords) - 1):
            x1, y1 = coords[i]
            x2, y2 = coords[i + 1]
            # segment length in km
            dx = (x2 - x1) * 111.32 * math.cos(lat_rad)
            dy = (y2 - y1) * 110.54
            seg_km = math.sqrt(dx**2 + dy**2)
            # midpoint pixel
            mx = (x1 + x2) / 2
            my_c = (y1 + y2) / 2
            col = int((mx - tf.c) / tf.a)
            row = int((my_c - tf.f) / tf.e)
            if 0 <= row < h and 0 <= col < w:
                river_raster[row, col] += seg_km

   
    kernel_px = max(3, int(1.0 / cellx_km))
    if kernel_px % 2 == 0:
        kernel_px += 1
    area_km2 = (kernel_px * cellx_km) * (kernel_px * celly_km)

    dd = uniform_filter(river_raster, size=kernel_px) * (kernel_px ** 2) / area_km2
    dd[np.isnan(dem)] = np.nan
    dd = dd.astype(np.float32)
    write_raster(out_path, dd, tf)
    log(f"  drainage_density range {np.nanmin(dd):.4f}–{np.nanmax(dd):.4f} km/km²")
    return out_path

# ── 7. Sample points + extract features ──
def sample_and_extract(dem_path):
    log("Sampling points and extracting features...")
    boundary = load_boundary()

    
    df_pos = pd.read_csv(os.path.join(DATASET, "titik_banjir_banyumas_geo.csv"))
    pts_pos = [(float(r["lon"]), float(r["lat"])) for _, r in df_pos.iterrows()]

    
    rng = np.random.default_rng(42)
    n_neg = len(pts_pos)  # balanced
    minx, miny, maxx, maxy = boundary.bounds
    lat_rad = math.radians(7.45)
    mx = 111320 * math.cos(lat_rad)
    my = 110540

    pos_arr = np.array([(y * my, x * mx) for x, y in pts_pos])
    pos_tree = cKDTree(pos_arr)

    pts_neg = []
    attempts = 0
    while len(pts_neg) < n_neg and attempts < 100000:
        x = rng.uniform(minx, maxx)
        y = rng.uniform(miny, maxy)
        p = Point(x, y)
        if not boundary.contains(p):
            attempts += 1
            continue
        d, _ = pos_tree.query([y * my, x * mx])
        if d < 2000:  # 2km buffer
            attempts += 1
            continue
        pts_neg.append((x, y))
        attempts += 1
    log(f"  {len(pts_pos)} positive, {len(pts_neg)} negative points")

    
    all_pts = [(x, y, 1) for x, y in pts_pos] + [(x, y, 0) for x, y in pts_neg]

    
    hujan = pd.read_csv(os.path.join(DATASET, "hujan_bmkg_tunggulwulung.csv"))
    
    rainfall_annual = 3921.6

    
    raster_files = {
        "elevation": os.path.join(PROCESSED, "dem_mosaic.tif"),
        "slope": os.path.join(PROCESSED, "slope.tif"),
        "twi": os.path.join(PROCESSED, "twi.tif"),
        "hand": os.path.join(PROCESSED, "hand.tif"),
        "dist_river": os.path.join(PROCESSED, "dist_river.tif"),
        "drainage_density": os.path.join(PROCESSED, "drainage_density.tif"),
    }
    
    wc_path = os.path.join(DATASET, "worldcover_banyumas.tif")

    raster_srcs = {k: rasterio.open(v) for k, v in raster_files.items()}
    wc_src = rasterio.open(wc_path)

    rows_data = []
    for x, y, label in all_pts:
        row = {"lon": x, "lat": y, "banjir": label, "rainfall": rainfall_annual}
        for name, src in raster_srcs.items():
            try:
                val = list(src.sample([(x, y)]))[0][0]
                row[name] = float(val) if not np.isnan(val) else np.nan
            except:
                row[name] = np.nan

        try:
            val = list(wc_src.sample([(x, y)]))[0][0]
            row["landcover"] = int(val)
        except:
            row["landcover"] = 0
        rows_data.append(row)

    for src in raster_srcs.values():
        src.close()
    wc_src.close()

    df = pd.DataFrame(rows_data)
    
    feat_cols = ["elevation", "slope", "twi", "hand", "dist_river", "drainage_density", "rainfall", "landcover"]
    before = len(df)
    df = df.dropna(subset=feat_cols).reset_index(drop=True)
    log(f"  {before} → {len(df)} after dropping NaN")

    return df, feat_cols

# ── 8. Spatial Block Split ──
def spatial_block_split(df, block_size=0.05, test_frac=0.2, seed=42):
    log("Spatial block split...")
    df = df.copy()
    df["block_x"] = (df["lon"] / block_size).astype(int)
    df["block_y"] = (df["lat"] / block_size).astype(int)
    df["block_id"] = df["block_x"].astype(str) + "_" + df["block_y"].astype(str)

    blocks = np.array(df["block_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(blocks)

    n_test = max(1, int(len(blocks) * test_frac))
    test_blocks = set(blocks[:n_test])
    train_blocks = set(blocks[n_test:])

    df_train = df[df["block_id"].isin(train_blocks)].drop(columns=["block_x", "block_y", "block_id"])
    df_test = df[df["block_id"].isin(test_blocks)].drop(columns=["block_x", "block_y", "block_id"])
    log(f"  {len(train_blocks)} train blocks ({len(df_train)} pts), {len(test_blocks)} test blocks ({len(df_test)} pts)")
    log(f"  train banjir: {df_train['banjir'].sum()}/{len(df_train)}, test banjir: {df_test['banjir'].sum()}/{len(df_test)}")
    return df_train, df_test

def main():
    t0 = time.time()
    log("=== START PREPROCESSING ===")

    dem_path = mosaic_clip()
    slope_path = calc_slope(dem_path)
    twi_path = calc_twi(dem_path, slope_path)
    hand_path = calc_hand(dem_path)
    dist_river_path = calc_dist_river(dem_path)
    dd_path = calc_drainage_density(dem_path)

    df, feat_cols = sample_and_extract(dem_path)
    df_train, df_test = spatial_block_split(df)

    train_path = os.path.join(PROCESSED, "features.csv")
    test_path = os.path.join(PROCESSED, "features_test.csv")
    df_train.to_csv(train_path, index=False)
    df_test.to_csv(test_path, index=False)
    log(f"Saved {train_path} ({len(df_train)} rows)")
    log(f"Saved {test_path} ({len(df_test)} rows)")
    log(f"=== DONE in {time.time()-t0:.0f}s ===")

if __name__ == "__main__":
    main()
