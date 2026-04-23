"""
analyze_residuals.py

Laedt dieselbe Pipeline wie main.py, macht CV-Predictions, und analysiert
wo das Modell versagt. Kein Eingriff in main.py.

Nutzt nur numpy + sklearn (keine matplotlib-Abhaengigkeit - alles text-based).
"""

from utils import load_config, load_dataset, IMAGE_SIZE
import numpy as np

from sklearn.model_selection import KFold
from main import extract_features, BaggedStackedHGB


def analyze():
    config = load_config()

    # Daten + Features laden (wie main.py)
    images, distances = load_dataset(config)
    use_rgb = config.get("load_rgb", False)
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]
    features = extract_features(images, image_size, use_rgb=use_rgb)

    y = np.asarray(distances)
    print(f"[INFO]: {len(y)} samples, {features.shape[1]} features")

    # ---------- Target-Verteilung ----------
    print("\n" + "=" * 70)
    print("TARGET-VERTEILUNG (distances in m)")
    print("=" * 70)
    print(f"  min    : {y.min():.3f}")
    print(f"  25%    : {np.percentile(y, 25):.3f}")
    print(f"  median : {np.median(y):.3f}")
    print(f"  mean   : {y.mean():.3f}")
    print(f"  75%    : {np.percentile(y, 75):.3f}")
    print(f"  max    : {y.max():.3f}")
    print(f"  std    : {y.std():.3f}")
    # Skewness manuell
    skew = ((y - y.mean()) ** 3).mean() / (y.std() ** 3 + 1e-12)
    print(f"  skew   : {skew:+.3f}  (0=symmetrisch, >0=Tail rechts, <0=Tail links)")

    # Histogram (text-based)
    print("\n  Histogram (10 Bins):")
    counts, edges = np.histogram(y, bins=10)
    max_count = counts.max()
    for i in range(len(counts)):
        bar_len = int(40 * counts[i] / max_count)
        print(f"  {edges[i]:.2f}-{edges[i+1]:.2f}: {'#' * bar_len} ({counts[i]})")

    # ---------- CV-Predictions sammeln ----------
    print("\n" + "=" * 70)
    print("CV-LAUF (sammelt Out-of-Fold Predictions)")
    print("=" * 70)

    kf = KFold(n_splits=config["cv_folds"], shuffle=True,
               random_state=config["random_seed"])

    oof_pred = np.zeros(len(y))

    for fold_i, (train_idx, val_idx) in enumerate(kf.split(features), 1):
        print(f"  Fold {fold_i}/{config['cv_folds']}... ", end="", flush=True)
        model = BaggedStackedHGB(
            hgb_params=config.get("model_params", {}),
            knn_params=config.get("knn_params", {}),
            et_params=config.get("et_params", {}),
            pca_components=config.get("pca_components", 50),
            seeds=config.get("bagging_seeds", [42, 123, 456]),
        )
        model.fit(features[train_idx], y[train_idx])
        oof_pred[val_idx] = model.predict(features[val_idx])
        fold_mae = np.abs(y[val_idx] - oof_pred[val_idx]).mean() * 100
        print(f"MAE = {fold_mae:.2f} cm")

    residuals = y - oof_pred   # positive: Modell hat unterschaetzt
    abs_err = np.abs(residuals)
    total_mae = abs_err.mean() * 100
    print(f"\n  Overall CV MAE: {total_mae:.2f} cm")

    # ---------- Residuen-Verteilung ----------
    print("\n" + "=" * 70)
    print("RESIDUEN-ANALYSE (y - y_pred, in m)")
    print("=" * 70)
    print(f"  mean     : {residuals.mean():+.4f}  (0 = unbiased)")
    print(f"  median   : {np.median(residuals):+.4f}")
    print(f"  std      : {residuals.std():.4f}")
    print(f"  25%      : {np.percentile(residuals, 25):+.4f}")
    print(f"  75%      : {np.percentile(residuals, 75):+.4f}")
    res_skew = ((residuals - residuals.mean()) ** 3).mean() / (residuals.std() ** 3 + 1e-12)
    print(f"  skew     : {res_skew:+.3f}")

    # ---------- Fehler nach Distanz-Bucket ----------
    print("\n" + "=" * 70)
    print("MAE NACH DISTANZ-BUCKET (Quintile)")
    print("=" * 70)
    quintile_edges = np.percentile(y, [0, 20, 40, 60, 80, 100])
    print(f"  Bucket range (m)     | N   | MAE (cm) | Bias (cm)")
    print(f"  " + "-" * 60)
    for i in range(5):
        lo, hi = quintile_edges[i], quintile_edges[i+1]
        mask = (y >= lo) & (y <= hi) if i == 4 else (y >= lo) & (y < hi)
        bucket_mae = abs_err[mask].mean() * 100
        bucket_bias = residuals[mask].mean() * 100
        print(f"  {lo:.2f} - {hi:.2f}          | {mask.sum():3d} | {bucket_mae:7.2f}  | {bucket_bias:+7.2f}")
    print(f"\n  Bias-Interpretation: positiv = Modell unterschaetzt in diesem Bucket,")
    print(f"                       negativ = Modell ueberschaetzt.")

    # ---------- Worst 10 Fehler ----------
    print("\n" + "=" * 70)
    print("TOP 10 WORST PREDICTIONS")
    print("=" * 70)
    worst_idx = np.argsort(abs_err)[-10:][::-1]
    print(f"  idx  | y_true  | y_pred  | residual (cm)")
    print(f"  " + "-" * 55)
    for idx in worst_idx:
        print(f"  {idx:4d} | {y[idx]:7.3f} | {oof_pred[idx]:7.3f} | {residuals[idx]*100:+8.2f}")

    # ---------- Korrelation Fehler vs Distanz ----------
    print("\n" + "=" * 70)
    print("SCHLUSSFOLGERUNGEN")
    print("=" * 70)

    # Ist der Fehler proportional zur Distanz (multiplikativ) oder konstant (additiv)?
    corr = np.corrcoef(y, abs_err)[0, 1]
    print(f"  Korrelation |Fehler| vs Distanz: {corr:+.3f}")
    if corr > 0.15:
        print("  -> Fehler waechst mit Distanz -> Log-Transform von y koennte helfen")
    elif corr < -0.15:
        print("  -> Fehler sinkt mit Distanz (ungewoehnlich)")
    else:
        print("  -> Fehler unabhaengig von Distanz -> Log-Transform wird nicht helfen")

    if abs(skew) > 0.5:
        print(f"  Target ist schief (skew={skew:+.2f}) -> Log-Transform potenziell hilfreich")
    else:
        print(f"  Target ist ~symmetrisch (skew={skew:+.2f}) -> Log-Transform wahrscheinlich nutzlos")

    # Buckets-Streuung
    bucket_maes = []
    for i in range(5):
        lo, hi = quintile_edges[i], quintile_edges[i+1]
        mask = (y >= lo) & (y <= hi) if i == 4 else (y >= lo) & (y < hi)
        bucket_maes.append(abs_err[mask].mean() * 100)
    max_min_ratio = max(bucket_maes) / min(bucket_maes)
    print(f"  Bucket-MAE Spanne: {min(bucket_maes):.2f} - {max(bucket_maes):.2f} cm")
    print(f"  Max/Min Ratio: {max_min_ratio:.2f}x")
    if max_min_ratio > 1.5:
        print("  -> Starke Ungleichheit zwischen Buckets -> gezielte Behandlung lohnt sich")
    else:
        print("  -> Fehler relativ gleichmaessig ueber Distanzen verteilt")


if __name__ == "__main__":
    analyze()