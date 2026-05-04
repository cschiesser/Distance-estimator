"""
V17 — Three angles combined to push under 9.5cm.

V16d gave us 9.91cm OOF with ETP-combined (raw PCA + engineered features).
The single best model `etp_comb_c_ds20_k2` was 10.06cm.

V17 combines:
    1. More ETP-combined variants (scales, k, preprocessing) — finds better
       individual models
    2. Bagged ETP-combined (10 forests per spec) — variance reduction
       lesson from v16b: bagging dropped 11.34 → 10.95 (-0.4cm)
    3. Enhanced engineered features:
       - Horizon line detection (vertical edge density)
       - Vanishing point features (left-right symmetry)
       - Multi-scale local variance (texture density at scale)
       - Gradient direction histograms (HOG-lite)
       - Inter-channel correlations for RGB

Goal: best individual model under 10cm AND blend under 9.5cm.
"""
from utils import load_config, load_dataset, load_test_dataset, save_results
import numpy as np
from pathlib import Path

from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA


# ─────────────────────────────────────────────────────────────────────────────
# Loading + preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def load_at_scale(data_dir, downsample, load_rgb=False):
    from utils import load_dataset
    cfg = {"data_dir": Path(data_dir), "load_rgb": load_rgb,
           "downsample_factor": downsample}
    imgs, dists = load_dataset(cfg)
    return np.asarray(imgs, dtype=float), np.asarray(dists, dtype=float)


def load_test_at_scale(data_dir, downsample, load_rgb=False):
    from utils import load_test_dataset
    cfg = {"data_dir": Path(data_dir), "load_rgb": load_rgb,
           "downsample_factor": downsample}
    return np.asarray(load_test_dataset(cfg), dtype=float)


def gamma_correction(imgs, gamma):
    return np.power(np.clip(imgs / 255.0, 0, 1), gamma) * 255.0


def clahe_approx(imgs, clip_limit=0.03, n_bins=256):
    n, nf = imgs.shape
    side  = int(round(nf ** 0.5))
    if side * side != nf:
        return imgs.copy()
    out = np.zeros_like(imgs)
    im2 = imgs.reshape(n, side, side)
    mh, mw = side // 2, side // 2
    for i in range(n):
        img = im2[i]; o = np.zeros_like(img)
        for r0, r1 in [(0, mh), (mh, side)]:
            for c0, c1 in [(0, mw), (mw, side)]:
                patch   = img[r0:r1, c0:c1].ravel()
                hist, _ = np.histogram(patch, bins=n_bins, range=(0, 255))
                cv      = max(1, int(clip_limit * patch.size))
                ex      = np.maximum(hist - cv, 0)
                hist    = np.minimum(hist, cv); hist += ex.sum() // n_bins
                cdf     = hist.cumsum(); nz = cdf[cdf > 0]
                if len(nz) == 0:
                    o[r0:r1, c0:c1] = img[r0:r1, c0:c1]; continue
                cn = (cdf - nz.min()) / (patch.size - nz.min() + 1e-9) * 255
                o[r0:r1, c0:c1] = cn[np.clip(
                    img[r0:r1, c0:c1].astype(int), 0, n_bins-1)]
        out[i] = o.ravel()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Engineered features — base set (from v16d) + NEW depth cues
# ─────────────────────────────────────────────────────────────────────────────

def build_engineered_features(imgs, n_channels=3):
    """V16d's base feature set."""
    n_samples = imgs.shape[0]
    nf = imgs.shape[1]
    px_per_ch = nf // n_channels
    side = int(round(px_per_ch ** 0.5))
    imgs_3d = imgs.reshape(n_samples, side, side, n_channels)
    feats = []

    feats.append(imgs.mean(axis=1, keepdims=True))
    feats.append(imgs.std(axis=1,  keepdims=True))
    for p in [5, 25, 50, 75, 95]:
        feats.append(np.percentile(imgs, p, axis=1, keepdims=True))

    rh = np.linspace(0, side, 4, dtype=int)
    rw = np.linspace(0, side, 4, dtype=int)
    for r in range(3):
        for c in range(3):
            patch = imgs_3d[:, rh[r]:rh[r+1], rw[c]:rw[c+1], :].reshape(n_samples, -1)
            feats.append(patch.mean(axis=1, keepdims=True))
            feats.append(patch.std(axis=1,  keepdims=True))
            feats.append(np.median(patch, axis=1, keepdims=True))

    mh = side // 2
    top  = imgs_3d[:, :mh, :, :].reshape(n_samples, -1).mean(axis=1)
    bot  = imgs_3d[:, mh:, :, :].reshape(n_samples, -1).mean(axis=1)
    feats.append((top - bot).reshape(-1, 1))
    feats.append((top / (bot + 1e-9)).reshape(-1, 1))

    qh, qw = side//4, side//4
    center  = imgs_3d[:, qh:side-qh, qw:side-qw, :].reshape(n_samples, -1).mean(axis=1)
    full    = imgs.mean(axis=1)
    feats.append((center - full).reshape(-1, 1))
    feats.append((center / (full + 1e-9)).reshape(-1, 1))

    for ch in range(n_channels):
        ch_img = imgs_3d[:, :, :, ch]
        gx = np.abs(np.diff(ch_img, axis=2))
        gy = np.abs(np.diff(ch_img, axis=1))
        feats.append(gx.reshape(n_samples, -1).mean(axis=1, keepdims=True))
        feats.append(gx.reshape(n_samples, -1).std(axis=1,  keepdims=True))
        feats.append(gy.reshape(n_samples, -1).mean(axis=1, keepdims=True))
        feats.append(gy.reshape(n_samples, -1).std(axis=1,  keepdims=True))

    hist_feats = np.zeros((n_samples, 32))
    for i in range(n_samples):
        h, _ = np.histogram(imgs[i], bins=32, range=(0, 255))
        hist_feats[i] = h / (h.sum() + 1e-9)
    feats.append(hist_feats)

    cy, cx = side / 2, side / 2
    yy, xx = np.mgrid[0:side, 0:side]
    dist_c = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    n_rings = 5
    edges = np.linspace(0, min(cy, cx), n_rings+1)
    for i in range(n_rings):
        mask = ((dist_c >= edges[i]) & (dist_c < edges[i+1]))
        ring = imgs_3d[:, mask, :]
        feats.append(ring.mean(axis=(1, 2)).reshape(n_samples, -1))
        feats.append(ring.std(axis=(1, 2)).reshape(n_samples, -1))

    return np.hstack(feats)


def build_enhanced_features(imgs, n_channels=3):
    """
    V16d's features PLUS new depth-cue features:

    1. Horizon line detection — find max vertical-edge row (where the
       horizon usually is). The horizon's row position is a strong depth cue.

    2. Left-right symmetry — vanishing point detection. Symmetric scenes
       (e.g., corridors looking forward) have specific depth patterns.

    3. Multi-scale local variance — texture density at 3 scales (close
       objects = textured, far = smooth).

    4. Gradient direction histograms (HOG-lite) — quantised edge directions
       per region. Strong feature for shape recognition.

    5. Inter-channel correlations (RGB only) — color gradients between
       channels reveal scene material/depth.
    """
    n_samples = imgs.shape[0]
    nf = imgs.shape[1]
    px_per_ch = nf // n_channels
    side = int(round(px_per_ch ** 0.5))
    imgs_3d = imgs.reshape(n_samples, side, side, n_channels)
    base_feats = build_engineered_features(imgs, n_channels)
    new_feats = []

    # ── 1. Horizon detection ──────────────────────────────────────────────
    # Average gradient magnitude per row (across all channels)
    # The horizon is typically the row with the strongest vertical edges
    gray_img = imgs_3d.mean(axis=3)  # (n, side, side)
    gy_full = np.abs(np.diff(gray_img, axis=1))  # (n, side-1, side)
    row_edges = gy_full.mean(axis=2)  # (n, side-1) — edge strength per row
    horizon_row = np.argmax(row_edges, axis=1)  # which row has max edge
    new_feats.append(horizon_row.reshape(-1, 1).astype(float))
    new_feats.append((horizon_row / max(1, side-1)).reshape(-1, 1))  # normalised

    # Brightness above vs below horizon
    above_horizon, below_horizon = [], []
    for i in range(n_samples):
        h = horizon_row[i]
        if h > 0:
            above_horizon.append(gray_img[i, :h, :].mean())
        else:
            above_horizon.append(gray_img[i].mean())
        if h < side - 1:
            below_horizon.append(gray_img[i, h:, :].mean())
        else:
            below_horizon.append(gray_img[i].mean())
    new_feats.append(np.array(above_horizon).reshape(-1, 1))
    new_feats.append(np.array(below_horizon).reshape(-1, 1))
    new_feats.append((np.array(above_horizon) -
                      np.array(below_horizon)).reshape(-1, 1))

    # ── 2. Left-right symmetry ────────────────────────────────────────────
    # Vanishing-point symmetry: compare left half mirrored vs right half
    mw = side // 2
    left  = imgs_3d[:, :, :mw, :]
    right = imgs_3d[:, :, mw:2*mw, :][:, :, ::-1, :]  # mirrored right
    sym_diff = np.abs(left - right).reshape(n_samples, -1).mean(axis=1)
    new_feats.append(sym_diff.reshape(-1, 1))
    sym_corr = np.zeros(n_samples)
    for i in range(n_samples):
        l_flat = left[i].ravel()
        r_flat = right[i].ravel()
        if l_flat.std() > 0 and r_flat.std() > 0:
            sym_corr[i] = np.corrcoef(l_flat, r_flat)[0, 1]
    new_feats.append(sym_corr.reshape(-1, 1))

    # ── 3. Multi-scale local variance ─────────────────────────────────────
    # Texture density at 3 scales — close objects more textured
    for scale in [2, 4, 8]:
        if scale >= side: continue
        patch_var = np.zeros(n_samples)
        n_blocks = side // scale
        if n_blocks == 0: continue
        for i in range(n_samples):
            block_vars = []
            for r in range(n_blocks):
                for c in range(n_blocks):
                    block = gray_img[i, r*scale:(r+1)*scale, c*scale:(c+1)*scale]
                    block_vars.append(block.var())
            patch_var[i] = np.mean(block_vars) if block_vars else 0
        new_feats.append(patch_var.reshape(-1, 1))

    # ── 4. HOG-lite gradient direction histogram ──────────────────────────
    # 4-bin gradient direction histogram per quadrant
    gx_full = np.diff(gray_img, axis=2)
    gy_full2 = np.diff(gray_img, axis=1)
    # Crop to common shape
    gx_c = gx_full[:, :-1, :]
    gy_c = gy_full2[:, :, :-1]
    angle = np.arctan2(gy_c, gx_c)  # radians, in [-π, π]
    mag = np.sqrt(gx_c**2 + gy_c**2)

    n_bins_hog = 4
    bin_edges = np.linspace(-np.pi, np.pi, n_bins_hog + 1)
    h2 = (side - 1) // 2
    for r0, r1 in [(0, h2), (h2, side-1)]:
        for c0, c1 in [(0, h2), (h2, side-1)]:
            quad_angle = angle[:, r0:r1, c0:c1]
            quad_mag   = mag[:,   r0:r1, c0:c1]
            for b in range(n_bins_hog):
                in_bin = ((quad_angle >= bin_edges[b]) &
                          (quad_angle < bin_edges[b+1]))
                # Magnitude-weighted count per bin per quadrant
                weighted = (quad_mag * in_bin).reshape(n_samples, -1).sum(axis=1)
                new_feats.append(weighted.reshape(-1, 1))

    # ── 5. Inter-channel correlations (RGB only) ──────────────────────────
    if n_channels == 3:
        for ch1 in range(3):
            for ch2 in range(ch1+1, 3):
                ch1_flat = imgs_3d[:, :, :, ch1].reshape(n_samples, -1)
                ch2_flat = imgs_3d[:, :, :, ch2].reshape(n_samples, -1)
                corr = np.zeros(n_samples)
                for i in range(n_samples):
                    if ch1_flat[i].std() > 0 and ch2_flat[i].std() > 0:
                        corr[i] = np.corrcoef(ch1_flat[i], ch2_flat[i])[0, 1]
                new_feats.append(corr.reshape(-1, 1))
                # Also: mean abs difference between channels
                diff = np.abs(ch1_flat - ch2_flat).mean(axis=1)
                new_feats.append(diff.reshape(-1, 1))

    return np.hstack([base_feats] + new_feats)


# ─────────────────────────────────────────────────────────────────────────────
# Tree-proximity
# ─────────────────────────────────────────────────────────────────────────────

def tree_proximity_predict(tree_model, X_tr, y_tr, X_te, k=3):
    tree_model.fit(X_tr, y_tr)
    tr_leaves = tree_model.apply(X_tr)
    te_leaves = tree_model.apply(X_te)
    knn = KNeighborsRegressor(n_neighbors=k, weights="distance",
                               metric="hamming")
    knn.fit(tr_leaves, y_tr)
    return knn.predict(te_leaves)


def bagged_etp_combined_predict(X_tr, y_tr, X_te, k=3, n_bags=10,
                                  base_seed=42, **et_params):
    preds = np.zeros((n_bags, len(X_te)))
    for b in range(n_bags):
        params = dict(et_params); params["random_state"] = base_seed + b * 1000
        et = ExtraTreesRegressor(**params)
        preds[b] = tree_proximity_predict(et, X_tr, y_tr, X_te, k=k)
    return preds.mean(axis=0)


# ─────────────────────────────────────────────────────────────────────────────
# OOF builders
# ─────────────────────────────────────────────────────────────────────────────

def bagged_knn_predict(X_tr, y_tr, X_te, n_bags=10, frac=0.8,
                        k=2, metric="manhattan", seed=42):
    rng   = np.random.RandomState(seed)
    n_sub = int(len(y_tr) * frac)
    preds = np.zeros((n_bags, len(X_te)))
    for b in range(n_bags):
        idx = rng.choice(len(y_tr), n_sub, replace=False)
        knn = KNeighborsRegressor(n_neighbors=k, weights="distance", metric=metric)
        knn.fit(X_tr[idx], y_tr[idx])
        preds[b] = knn.predict(X_te)
    return preds.mean(axis=0)


def multi_split_knn_oof(raw_data, y_data, preproc_fn, n_bags=10,
                         n_splits=5, k=2, metric="manhattan",
                         seeds=(42, 123, 7), label=""):
    oofs = []
    for seed in seeds:
        kf  = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y_data))
        for tr_f, val_f in kf.split(raw_data):
            sc  = RobustScaler(quantile_range=(25, 75))
            X_tr = sc.fit_transform(preproc_fn(raw_data[tr_f]))
            X_va = sc.transform(preproc_fn(raw_data[val_f]))
            oof[val_f] = bagged_knn_predict(X_tr, y_data[tr_f], X_va,
                                              n_bags=n_bags, k=k, metric=metric,
                                              seed=seed)
        oofs.append(oof)
    avg = np.mean(oofs, axis=0)
    print(f"  KNN  {label:55s} | OOF={mean_absolute_error(y_data, avg)*100:.2f}")
    return avg


def etp_combined_oof(raw_data, y_data, n_channels, preproc_fn=None,
                      enhanced=False, n_pca=80, k=3, n_bags=1,
                      n_splits=5, seeds=(42, 123, 7), label="", **et_params):
    """
    OOF for ETP-combined (PCA pixels + engineered features).

    enhanced: if True, use enhanced engineered features (with new depth cues)
    n_bags: if >1, bag multiple ETP forests with different seeds
    preproc_fn: optional preprocessing applied to raw before PCA
    """
    default = {"n_estimators": 500, "max_depth": None,
               "min_samples_leaf": 5, "max_features": 0.3,
               "n_jobs": -1}
    default.update(et_params)

    feature_fn = build_enhanced_features if enhanced else build_engineered_features

    oofs = []
    for seed in seeds:
        kf  = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        oof = np.zeros(len(y_data))
        for tr_f, val_f in kf.split(raw_data):
            # Apply preprocessing if given
            X_tr_raw = preproc_fn(raw_data[tr_f]) if preproc_fn else raw_data[tr_f]
            X_va_raw = preproc_fn(raw_data[val_f]) if preproc_fn else raw_data[val_f]

            # PCA pixels
            sc1 = RobustScaler(quantile_range=(25, 75))
            X_tr_pix = sc1.fit_transform(X_tr_raw)
            X_va_pix = sc1.transform(X_va_raw)
            n_comp = min(n_pca, X_tr_pix.shape[0]-1, X_tr_pix.shape[1])
            pca = PCA(n_components=n_comp, random_state=seed)
            X_tr_pca = pca.fit_transform(X_tr_pix)
            X_va_pca = pca.transform(X_va_pix)

            # Engineered features (built from preprocessed raw)
            X_tr_eng = feature_fn(X_tr_raw, n_channels)
            X_va_eng = feature_fn(X_va_raw, n_channels)
            sc2 = RobustScaler(quantile_range=(25, 75))
            X_tr_eng_s = sc2.fit_transform(X_tr_eng)
            X_va_eng_s = sc2.transform(X_va_eng)

            # Combine
            X_tr_combo = np.hstack([X_tr_pca, X_tr_eng_s])
            X_va_combo = np.hstack([X_va_pca, X_va_eng_s])

            if n_bags == 1:
                params = dict(default); params["random_state"] = seed
                et = ExtraTreesRegressor(**params)
                oof[val_f] = tree_proximity_predict(et, X_tr_combo,
                                                      y_data[tr_f], X_va_combo, k=k)
            else:
                oof[val_f] = bagged_etp_combined_predict(
                    X_tr_combo, y_data[tr_f], X_va_combo,
                    k=k, n_bags=n_bags, base_seed=seed, **default)
        oofs.append(oof)
    avg = np.mean(oofs, axis=0)
    suffix = f"bag{n_bags}" if n_bags > 1 else ""
    enh_str = "ENH" if enhanced else "BASE"
    print(f"  ETP-combo {enh_str:4} {suffix:>6} {label:40s} | "
          f"OOF={mean_absolute_error(y_data, avg)*100:.2f}")
    return avg


# ─────────────────────────────────────────────────────────────────────────────
# Refit functions for kaggle
# ─────────────────────────────────────────────────────────────────────────────

def refit_knn(raw_full, test_imgs, y_full, preproc_fn,
               n_bags=15, k=2, metric="manhattan"):
    sc  = RobustScaler(quantile_range=(25, 75))
    X_f = sc.fit_transform(preproc_fn(raw_full))
    X_t = sc.transform(preproc_fn(test_imgs))
    return bagged_knn_predict(X_f, y_full, X_t, n_bags=n_bags, k=k, metric=metric)


def refit_etp_combined(raw_full, test_imgs, y_full, n_channels,
                         preproc_fn=None, enhanced=False, n_pca=80, k=3,
                         n_bags=1, **et_params):
    default = {"n_estimators": 500, "max_depth": None,
               "min_samples_leaf": 5, "max_features": 0.3,
               "n_jobs": -1, "random_state": 42}
    default.update(et_params)
    feature_fn = build_enhanced_features if enhanced else build_engineered_features

    raw_full_p = preproc_fn(raw_full) if preproc_fn else raw_full
    test_p     = preproc_fn(test_imgs) if preproc_fn else test_imgs

    sc1 = RobustScaler(quantile_range=(25, 75))
    X_f_pix = sc1.fit_transform(raw_full_p)
    X_t_pix = sc1.transform(test_p)
    n_comp = min(n_pca, X_f_pix.shape[0]-1, X_f_pix.shape[1])
    pca = PCA(n_components=n_comp, random_state=42)
    X_f_pca = pca.fit_transform(X_f_pix)
    X_t_pca = pca.transform(X_t_pix)

    X_f_eng = feature_fn(raw_full_p, n_channels)
    X_t_eng = feature_fn(test_p, n_channels)
    sc2 = RobustScaler(quantile_range=(25, 75))
    X_f_eng_s = sc2.fit_transform(X_f_eng)
    X_t_eng_s = sc2.transform(X_t_eng)

    X_f_combo = np.hstack([X_f_pca, X_f_eng_s])
    X_t_combo = np.hstack([X_t_pca, X_t_eng_s])

    if n_bags == 1:
        return tree_proximity_predict(ExtraTreesRegressor(**default),
                                        X_f_combo, y_full, X_t_combo, k=k)
    else:
        params = dict(default); params.pop("random_state", None)
        return bagged_etp_combined_predict(X_f_combo, y_full, X_t_combo,
                                              k=k, n_bags=n_bags,
                                              base_seed=42, **params)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from utils import load_config
    config   = load_config()
    data_dir = config["data_dir"]

    print("[INFO]: Loading dataset …")
    _, distances = load_at_scale(data_dir, downsample=5)
    z_mask  = np.abs((distances - distances.mean()) / distances.std()) < 3.0
    y_data  = distances[z_mask]
    print(f"[INFO]: After zscore cleaning: {z_mask.sum()} samples")

    print("\n[INFO]: Loading images …")
    raw_g = {}
    for ds in [30, 25, 20, 15, 12, 10]:
        imgs, _ = load_at_scale(data_dir, ds)
        raw_g[ds] = imgs[z_mask]
    raw_c = {}
    for ds in [30, 25, 20, 15, 12, 10]:
        imgs, _ = load_at_scale(data_dir, ds, load_rgb=True)
        raw_c[ds] = imgs[z_mask]

    SEEDS = (42, 123, 7)
    all_oof = {}
    all_specs = {}

    # ── 1. KNN baseline ───────────────────────────────────────────────────────
    print(f"\n[INFO]: ── Proven KNN ──")
    knn_specs = [
        ("c_raw_ds30",     lambda x: x.copy(),                 30, True),
        ("c_raw_ds25",     lambda x: x.copy(),                 25, True),
        ("c_gamma05_ds20", lambda x: gamma_correction(x, 0.5), 20, True),
        ("g_raw_ds20",     lambda x: x.copy(),                 20, False),
    ]
    for name, fn, ds, rgb in knn_specs:
        oof = multi_split_knn_oof((raw_c if rgb else raw_g)[ds], y_data, fn,
                                    seeds=SEEDS, label=name)
        all_oof[name]   = oof
        all_specs[name] = ("knn", fn, ds, rgb)

    # ── 2. ANGLE 1: More ETP-combined variants (BASE features) ───────────────
    print(f"\n[INFO]: ── ANGLE 1: More ETP-combined variants ──")
    angle1_specs = [
        # v16d winners (baseline for comparison)
        ("comb_c_ds20_k2",        20, True,  None,  False, 80, 2, 1, {}),
        ("comb_c_ds15",           15, True,  None,  False, 80, 3, 1, {}),
        ("comb_c_ds20",           20, True,  None,  False, 80, 3, 1, {}),
        # NEW: more scales
        ("comb_c_ds10_k2",        10, True,  None,  False, 80, 2, 1, {}),
        ("comb_c_ds12_k2",        12, True,  None,  False, 80, 2, 1, {}),
        ("comb_c_ds25_k2",        25, True,  None,  False, 80, 2, 1, {}),
        # NEW: more k values
        ("comb_c_ds20_k4",        20, True,  None,  False, 80, 4, 1, {}),
        ("comb_c_ds15_k2",        15, True,  None,  False, 80, 2, 1, {}),
        # NEW: with preprocessing
        ("comb_c_gamma05_ds20_k2", 20, True,  lambda x: gamma_correction(x, 0.5),
         False, 80, 2, 1, {}),
        ("comb_c_gamma07_ds20_k2", 20, True,  lambda x: gamma_correction(x, 0.7),
         False, 80, 2, 1, {}),
        ("comb_c_clahe_ds15",     15, True,  lambda x: clahe_approx(x, 0.03),
         False, 80, 3, 1, {}),
        # NEW: deeper trees
        ("comb_c_ds20_deep_k2",   20, True,  None,  False, 80, 2, 1,
         {"min_samples_leaf": 2, "max_features": 0.5}),
        # NEW: different PCA depths
        ("comb_c_ds20_pca60_k2",  20, True,  None,  False, 60, 2, 1, {}),
        ("comb_c_ds20_pca100_k2", 20, True,  None,  False, 100, 2, 1, {}),
    ]
    for name, ds, rgb, fn, enh, n_pca, k, n_bags, params in angle1_specs:
        n_ch = 3 if rgb else 1
        raw_src = raw_c[ds] if rgb else raw_g[ds]
        oof = etp_combined_oof(raw_src, y_data, n_ch, preproc_fn=fn,
                                 enhanced=enh, n_pca=n_pca, k=k, n_bags=n_bags,
                                 seeds=SEEDS, label=name, **params)
        all_oof[name]   = oof
        all_specs[name] = ("etp_comb", ds, rgb, n_ch, fn, enh, n_pca, k, n_bags, params)

    # ── 3. ANGLE 2: Bagged ETP-combined ──────────────────────────────────────
    print(f"\n[INFO]: ── ANGLE 2: Bagged ETP-combined (10 forests/spec) ──")
    angle2_specs = [
        ("combb10_c_ds20_k2",       20, True,  None,  False, 80, 2, 10, {}),
        ("combb10_c_ds15_k2",       15, True,  None,  False, 80, 2, 10, {}),
        ("combb10_c_ds20",          20, True,  None,  False, 80, 3, 10, {}),
        ("combb10_c_ds15",          15, True,  None,  False, 80, 3, 10, {}),
        ("combb10_c_gamma05_ds20",  20, True,  lambda x: gamma_correction(x, 0.5),
         False, 80, 2, 10, {}),
    ]
    for name, ds, rgb, fn, enh, n_pca, k, n_bags, params in angle2_specs:
        n_ch = 3 if rgb else 1
        raw_src = raw_c[ds] if rgb else raw_g[ds]
        oof = etp_combined_oof(raw_src, y_data, n_ch, preproc_fn=fn,
                                 enhanced=enh, n_pca=n_pca, k=k, n_bags=n_bags,
                                 seeds=SEEDS, label=name, **params)
        all_oof[name]   = oof
        all_specs[name] = ("etp_comb", ds, rgb, n_ch, fn, enh, n_pca, k, n_bags, params)

    # ── 4. ANGLE 3: Enhanced engineered features ─────────────────────────────
    print(f"\n[INFO]: ── ANGLE 3: ENHANCED engineered features (horizon, HOG, …) ──")
    angle3_specs = [
        ("combE_c_ds20_k2",        20, True,  None,  True, 80, 2, 1, {}),
        ("combE_c_ds15_k2",        15, True,  None,  True, 80, 2, 1, {}),
        ("combE_c_ds20",           20, True,  None,  True, 80, 3, 1, {}),
        ("combE_c_ds15",           15, True,  None,  True, 80, 3, 1, {}),
        ("combE_c_ds30_k2",        30, True,  None,  True, 80, 2, 1, {}),
        # Bagged + Enhanced
        ("combEb10_c_ds20_k2",     20, True,  None,  True, 80, 2, 10, {}),
        ("combEb10_c_ds15_k2",     15, True,  None,  True, 80, 2, 10, {}),
        # Enhanced + preprocessing
        ("combE_c_gamma05_ds20_k2", 20, True, lambda x: gamma_correction(x, 0.5),
         True, 80, 2, 1, {}),
    ]
    for name, ds, rgb, fn, enh, n_pca, k, n_bags, params in angle3_specs:
        n_ch = 3 if rgb else 1
        raw_src = raw_c[ds] if rgb else raw_g[ds]
        oof = etp_combined_oof(raw_src, y_data, n_ch, preproc_fn=fn,
                                 enhanced=enh, n_pca=n_pca, k=k, n_bags=n_bags,
                                 seeds=SEEDS, label=name, **params)
        all_oof[name]   = oof
        all_specs[name] = ("etp_comb", ds, rgb, n_ch, fn, enh, n_pca, k, n_bags, params)

    # ── 5. Sort + analyse ─────────────────────────────────────────────────────
    print(f"\n[INFO]: ── All {len(all_oof)} models by OOF MAE ──")
    sorted_models = sorted(all_oof.items(),
                            key=lambda x: mean_absolute_error(y_data, x[1]))
    for name, oof in sorted_models:
        mae = mean_absolute_error(y_data, oof)
        print(f"  {name:55s} | OOF={mae*100:.2f}")

    # ── 6. Pairwise blend ─────────────────────────────────────────────────────
    print(f"\n[INFO]: ── Pairwise blend ──")
    names = list(all_oof.keys())
    pair_results = []
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            best_a, best_m = 0.5, np.inf
            for alpha in np.arange(0.0, 1.01, 0.01):
                m = mean_absolute_error(y_data,
                    alpha * all_oof[names[i]] + (1-alpha) * all_oof[names[j]])
                if m < best_m:
                    best_m, best_a = m, alpha
            pair_results.append((names[i], names[j], best_a, best_m))
    pair_results.sort(key=lambda x: x[3])
    print("  Top 15 pairs:")
    for n1, n2, a, mae in pair_results[:15]:
        print(f"    {n1} ({a:.2f}) + {n2} ({1-a:.2f}) → {mae*100:.2f}")
    n1_b, n2_b, alpha_b, mae_pair = pair_results[0]
    oof_pair = alpha_b * all_oof[n1_b] + (1-alpha_b) * all_oof[n2_b]

    # ── 7. 3-way blend ────────────────────────────────────────────────────────
    top_pair_models = list(set(
        [n for tup in pair_results[:10] for n in (tup[0], tup[1])]))
    print(f"\n[INFO]: 3-way among {len(top_pair_models)} models")
    best_3way_mae = np.inf
    best_3way_oof = None
    best_3way_data = None
    for i in range(len(top_pair_models)):
        for j in range(i+1, len(top_pair_models)):
            for k in range(j+1, len(top_pair_models)):
                n1, n2, n3 = top_pair_models[i], top_pair_models[j], top_pair_models[k]
                for w0 in np.arange(0.0, 1.01, 0.05):
                    for w1 in np.arange(0.0, 1.01-w0, 0.05):
                        w2 = round(1-w0-w1, 10)
                        if w2 < 0: continue
                        oof_b = w0*all_oof[n1] + w1*all_oof[n2] + w2*all_oof[n3]
                        mae = mean_absolute_error(y_data, oof_b)
                        if mae < best_3way_mae:
                            best_3way_mae = mae
                            best_3way_oof = oof_b
                            best_3way_data = (n1, n2, n3, w0, w1, w2)
    n1, n2, n3, w0, w1, w2 = best_3way_data
    print(f"  Best 3-way: {n1}({w0:.2f})+{n2}({w1:.2f})+{n3}({w2:.2f}) → OOF={best_3way_mae*100:.2f}")

    # ── 8. 4-way blend ────────────────────────────────────────────────────────
    top_pair_models_8 = top_pair_models[:8] if len(top_pair_models) > 8 else top_pair_models
    print(f"\n[INFO]: 4-way among {len(top_pair_models_8)} models")
    best_4way_mae = np.inf
    best_4way_oof = None
    best_4way_data = None
    for i in range(len(top_pair_models_8)):
        for j in range(i+1, len(top_pair_models_8)):
            for k in range(j+1, len(top_pair_models_8)):
                for l in range(k+1, len(top_pair_models_8)):
                    n1, n2, n3, n4 = (top_pair_models_8[i], top_pair_models_8[j],
                                       top_pair_models_8[k], top_pair_models_8[l])
                    for w0 in np.arange(0.0, 1.01, 0.1):
                        for w1 in np.arange(0.0, 1.01-w0, 0.1):
                            for w2 in np.arange(0.0, 1.01-w0-w1, 0.1):
                                w3 = round(1-w0-w1-w2, 10)
                                if w3 < 0: continue
                                oof_b = (w0*all_oof[n1] + w1*all_oof[n2] +
                                         w2*all_oof[n3] + w3*all_oof[n4])
                                mae = mean_absolute_error(y_data, oof_b)
                                if mae < best_4way_mae:
                                    best_4way_mae = mae
                                    best_4way_oof = oof_b
                                    best_4way_data = (n1, n2, n3, n4, w0, w1, w2, w3)
    n1, n2, n3, n4, w0, w1, w2, w3 = best_4way_data
    print(f"  Best 4-way → OOF={best_4way_mae*100:.2f}")

    # ── 9. Ridge meta ─────────────────────────────────────────────────────────
    top15 = [n for n, _ in sorted_models[:15]]
    oof_mat = np.column_stack([all_oof[n] for n in top15])
    print("\n[INFO]: Ridge meta-learner …")
    best_ridge_mae = np.inf
    best_alpha = None
    best_ridge_oof = None
    for alpha in [0.5, 1.0, 2.5, 5.0, 10.0]:
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        ridge_oof = np.zeros(len(y_data))
        for tr_f, val_f in kf.split(oof_mat):
            mf = Ridge(alpha=alpha); mf.fit(oof_mat[tr_f], y_data[tr_f])
            ridge_oof[val_f] = mf.predict(oof_mat[val_f])
        mae = mean_absolute_error(y_data, ridge_oof)
        print(f"  Ridge alpha={alpha:6.2f} | OOF={mae*100:.2f}")
        if mae < best_ridge_mae:
            best_ridge_mae = mae; best_alpha = alpha; best_ridge_oof = ridge_oof

    # ── 10. Combos ────────────────────────────────────────────────────────────
    combo_results = {}
    for blend_name, blend_oof in [("pair", oof_pair), ("3way", best_3way_oof),
                                    ("4way", best_4way_oof)]:
        best = np.inf; best_w = 0.5
        for w in np.arange(0.0, 1.01, 0.01):
            mae = mean_absolute_error(y_data, w * best_ridge_oof + (1-w) * blend_oof)
            if mae < best: best, best_w = mae, w
        combo_results[f"Ridge({best_w:.2f})+{blend_name}"] = (best, best_w, blend_name)
        print(f"  Ridge({best_w:.2f})+{blend_name}({1-best_w:.2f}) → OOF={best*100:.2f}")

    candidates = {"Best pair": mae_pair, "Best 3-way": best_3way_mae,
                  "Best 4-way": best_4way_mae, "Ridge meta": best_ridge_mae}
    for name, (mae, _, _) in combo_results.items():
        candidates[name] = mae
    for n in top15:
        candidates[n] = mean_absolute_error(y_data, all_oof[n])

    winner = min(candidates, key=candidates.get)
    print(f"\n[INFO]: ══ TOP 10 ══")
    for label, mae in sorted(candidates.items(), key=lambda x: x[1])[:10]:
        marker = " ← WINNER" if label == winner else ""
        print(f"  {label:55s} → OOF={mae*100:.2f}{marker}")
    print(f"\n[RESULT]: Best OOF = {candidates[winner]*100:.2f} (winner: {winner})")

    # ── 11. Kaggle submission ─────────────────────────────────────────────────
    print(f"\n[INFO]: Building submission …")
    test_kag = {}
    for name, spec in all_specs.items():
        kind = spec[0]
        try:
            if kind == "knn":
                _, fn, ds, rgb = spec
                raw_src  = raw_c[ds] if rgb else raw_g[ds]
                test_src = load_test_at_scale(data_dir, ds, load_rgb=rgb)
                test_kag[name] = refit_knn(raw_src, test_src, y_data, fn, n_bags=15)
            elif kind == "etp_comb":
                _, ds, rgb, n_ch, fn, enh, n_pca, k, n_bags, params = spec
                raw_src  = raw_c[ds] if rgb else raw_g[ds]
                test_src = load_test_at_scale(data_dir, ds, load_rgb=rgb)
                test_kag[name] = refit_etp_combined(
                    raw_src, test_src, y_data, n_ch,
                    preproc_fn=fn, enhanced=enh, n_pca=n_pca, k=k,
                    n_bags=n_bags, **params)
            print(f"  [kag] {name} done")
        except Exception as e:
            print(f"  [kag] {name} FAILED: {e}")

    n_kag = len(list(test_kag.values())[0])
    kag_pair = (alpha_b * test_kag.get(n1_b, np.zeros(n_kag)) +
                (1-alpha_b) * test_kag.get(n2_b, np.zeros(n_kag)))
    n1, n2, n3, w0, w1, w2 = best_3way_data
    kag_3way = (w0 * test_kag.get(n1, np.zeros(n_kag)) +
                w1 * test_kag.get(n2, np.zeros(n_kag)) +
                w2 * test_kag.get(n3, np.zeros(n_kag)))
    n1, n2, n3, n4, w0, w1, w2, w3 = best_4way_data
    kag_4way = (w0 * test_kag.get(n1, np.zeros(n_kag)) +
                w1 * test_kag.get(n2, np.zeros(n_kag)) +
                w2 * test_kag.get(n3, np.zeros(n_kag)) +
                w3 * test_kag.get(n4, np.zeros(n_kag)))
    ridge_final = Ridge(alpha=best_alpha)
    ridge_final.fit(oof_mat, y_data)
    kag_mat = np.column_stack([test_kag.get(n, np.zeros(n_kag)) for n in top15])
    kag_ridge = ridge_final.predict(kag_mat)

    blend_kag = {"pair": kag_pair, "3way": kag_3way, "4way": kag_4way}
    for combo_name, (mae, w, blend_name) in combo_results.items():
        test_kag[combo_name] = w * kag_ridge + (1-w) * blend_kag[blend_name]
    test_kag["Best pair"] = kag_pair
    test_kag["Best 3-way"] = kag_3way
    test_kag["Best 4-way"] = kag_4way
    test_kag["Ridge meta"] = kag_ridge

    final_pred = test_kag.get(winner, kag_pair)
    save_results(final_pred)
    print(f"\n[INFO]: Submission saved with '{winner}' (~{candidates[winner]*100:.2f}cm)")