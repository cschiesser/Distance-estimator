"""
tune_hgb.py

Sucht die besten HGB-Hyperparameter für die Stacking-Strategie.
Nutzt Randomized Search über den Parameter-Space (schneller als Full Grid).
kNN-Params bleiben fix (aus config.yaml, idealerweise n_neighbors=3, weights='distance').
"""

from utils import load_config, load_dataset, IMAGE_SIZE
import numpy as np
import time

from sklearn.model_selection import KFold, cross_val_score

from main import extract_features, KnnStackedHGB


def sample_hgb_config(rng):
    """Sampled einen zufälligen Parameter-Satz aus sinnvollen Verteilungen."""
    return {
        # max_iter: wie viele Bäume. Mehr = mehr Kapazität, aber auch mehr Overfit
        "max_iter": int(rng.choice([400, 600, 800, 1000, 1200])),

        # learning_rate: wie stark jeder Baum korrigiert. Kleiner = mehr Bäume nötig, aber stabiler
        "learning_rate": float(rng.choice([0.01, 0.02, 0.03, 0.05, 0.08])),

        # max_depth: Tiefe pro Baum. Tiefer = mehr Interaktionen, mehr Overfit-Risiko
        # 90% der Zeit einen konkreten Wert, 10% None (= keine Tiefenbeschränkung)
        "max_depth": int(rng.choice([4, 6, 8, 10])) if rng.random() < 0.9 else None,

        # l2_regularization: straft komplexe Splits. Höher = simpler, weniger Overfit
        "l2_regularization": float(rng.choice([0.0, 0.1, 0.5, 1.0, 2.0, 5.0])),

        # min_samples_leaf: minimale Samples pro Blatt. Höher = robuster, weniger Overfit
        "min_samples_leaf": int(rng.choice([10, 20, 30, 50, 80])),

        # loss bleibt absolute_error (MAE-optimiert) - das ist schon korrekt
        "loss": "absolute_error",
    }


def tune(n_trials=30):
    config = load_config()

    # Daten laden
    images, distances = load_dataset(config)
    use_rgb = config.get("load_rgb", False)
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]
    features = extract_features(images, image_size, use_rgb=use_rgb)

    print(f"[INFO]: Dataset {len(images)} samples, features {features.shape[1]}, RGB={use_rgb}")
    print(f"[INFO]: Tuning HGB within Stacking. kNN-Params bleiben fix aus config.")
    print(f"[INFO]: kNN config: {config.get('knn_params', {})}")
    print(f"[INFO]: Running {n_trials} random trials...\n")

    knn_params = config.get("knn_params", {"n_neighbors": 3, "weights": "distance"})
    seed = config["random_seed"]

    # Baseline: aktuelle config HGB-Params
    baseline_params = config.get("model_params", {})

    # CV setup
    kf = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=seed)

    rng = np.random.default_rng(seed)
    results = []
    t_start = time.time()

    # Erst Baseline messen zum Vergleich
    print("[INFO]: Baseline run with current config.yaml HGB params...")
    baseline_model = KnnStackedHGB(
        hgb_params=baseline_params, knn_params=knn_params, seed=seed,
    )
    t0 = time.time()
    baseline_scores = cross_val_score(
        baseline_model, features, distances, cv=kf,
        scoring="neg_mean_absolute_error", n_jobs=-1,
    )
    baseline_mae = (-baseline_scores * 100).mean()
    baseline_std = (-baseline_scores * 100).std()
    print(f"[BASELINE] CV MAE = {baseline_mae:.3f} +/- {baseline_std:.3f} cm "
          f"({time.time()-t0:.1f}s)")
    print(f"[BASELINE] Params: {baseline_params}\n")

    # Random search
    best_so_far = baseline_mae
    for trial in range(1, n_trials + 1):
        params = sample_hgb_config(rng)

        model = KnnStackedHGB(
            hgb_params=params, knn_params=knn_params, seed=seed,
        )

        t0 = time.time()
        scores = cross_val_score(
            model, features, distances, cv=kf,
            scoring="neg_mean_absolute_error", n_jobs=-1,
        )
        elapsed = time.time() - t0
        mae = (-scores * 100).mean()
        std = (-scores * 100).std()
        results.append((params, mae, std))

        marker = ""
        if mae < best_so_far:
            marker = " <-- NEW BEST"
            best_so_far = mae

        # Compact print
        short = (f"max_iter={params['max_iter']:4d} "
                 f"lr={params['learning_rate']:.2f} "
                 f"depth={params['max_depth']} "
                 f"l2={params['l2_regularization']:.1f} "
                 f"leaf={params['min_samples_leaf']:3d}")
        print(f"[{trial:2d}/{n_trials}] {short} -> {mae:.3f} +/- {std:.3f} ({elapsed:.1f}s){marker}")

    total_time = time.time() - t_start
    print(f"\n[INFO]: Total tuning time: {total_time:.1f}s")
    print(f"[INFO]: Baseline was: {baseline_mae:.3f} cm\n")

    # Top 5 ausgeben
    results.sort(key=lambda r: r[1])
    print("=" * 90)
    print("TOP 5 CONFIGS:")
    print("=" * 90)
    for rank, (params, mae, std) in enumerate(results[:5], 1):
        print(f"\n#{rank}  CV MAE = {mae:.3f} +/- {std:.3f} cm")
        for k, v in params.items():
            print(f"      {k}: {v}")

    best_params, best_mae, _ = results[0]
    improvement = baseline_mae - best_mae
    print("\n" + "=" * 90)
    print(f"Best:     {best_mae:.3f} cm")
    print(f"Baseline: {baseline_mae:.3f} cm")
    print(f"Delta:    {improvement:+.3f} cm ({improvement/baseline_mae*100:+.1f}%)")
    print("=" * 90)
    print("\nIn config.yaml eintragen:")
    print("model_params:")
    for k, v in best_params.items():
        if isinstance(v, str):
            print(f'  {k}: "{v}"')
        else:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    tune(n_trials=30)