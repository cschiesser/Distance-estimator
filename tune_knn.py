"""
tune_knn.py

Sucht die besten kNN-Hyperparameter für die Stacking-Strategie.
Iteriert über (n_neighbors x weights x metric) und misst CV-MAE.
Nutzt dieselbe Feature-Extraktion wie main.py.
"""

from utils import load_config, load_dataset, IMAGE_SIZE
import numpy as np
import time

from sklearn.model_selection import KFold, cross_val_score

# Import aus main.py wiederverwenden (keine Duplikate)
from main import extract_features, KnnStackedHGB


def tune():
    config = load_config()

    # Daten + Features laden (genau wie in main.py)
    images, distances = load_dataset(config)
    use_rgb = config.get("load_rgb", False)
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]
    features = extract_features(images, image_size, use_rgb=use_rgb)

    print(f"[INFO]: Dataset {len(images)} samples, features {features.shape[1]}, RGB={use_rgb}")
    print(f"[INFO]: Tuning kNN within Stacking. HGB-Params bleiben fix aus config.\n")

    hgb_params = config.get("model_params", {})
    seed = config["random_seed"]

    # ---------- Search Space ----------
    n_neighbors_list = [3, 5, 7, 10, 15, 20, 30]
    weights_list = ["uniform", "distance"]
    metric_list = ["minkowski"]  # minkowski p=2 = euclidean; optional "manhattan" hinzufügen

    # ---------- CV-Setup ----------
    kf = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=seed)

    results = []
    total = len(n_neighbors_list) * len(weights_list) * len(metric_list)
    i = 0
    t_start = time.time()

    for k in n_neighbors_list:
        for w in weights_list:
            for m in metric_list:
                i += 1
                knn_params = {"n_neighbors": k, "weights": w}

                model = KnnStackedHGB(
                    hgb_params=hgb_params,
                    knn_params=knn_params,
                    seed=seed,
                )

                t0 = time.time()
                scores = cross_val_score(
                    model, features, distances, cv=kf,
                    scoring="neg_mean_absolute_error", n_jobs=-1,
                )
                elapsed = time.time() - t0
                mae_cm = -scores * 100

                mean_mae = mae_cm.mean()
                std_mae = mae_cm.std()
                results.append((k, w, m, mean_mae, std_mae))
                print(f"[{i:2d}/{total}] k={k:3d} w={w:<9s} m={m:<10s} "
                      f"-> CV MAE = {mean_mae:.3f} +/- {std_mae:.3f} cm  ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    print(f"\n[INFO]: Total tuning time: {total_time:.1f}s\n")

    # ---------- Ergebnisse sortieren ----------
    results.sort(key=lambda r: r[3])  # nach mean_mae aufsteigend
    print("=" * 70)
    print("RANKING (beste zuerst):")
    print("=" * 70)
    print(f"{'Rank':<6}{'k':<6}{'weights':<12}{'metric':<12}{'MAE':<10}{'Std':<8}")
    print("-" * 70)
    for rank, (k, w, m, mae, std) in enumerate(results, 1):
        marker = " <-- BEST" if rank == 1 else ""
        print(f"{rank:<6}{k:<6}{w:<12}{m:<12}{mae:<10.3f}{std:<8.3f}{marker}")

    best_k, best_w, best_m, best_mae, _ = results[0]
    print("\n" + "=" * 70)
    print(f"Best config: n_neighbors={best_k}, weights='{best_w}', metric='{best_m}'")
    print(f"Best CV MAE: {best_mae:.3f} cm")
    print("=" * 70)
    print("\nIn config.yaml eintragen:")
    print(f"knn_params:")
    print(f"  n_neighbors: {best_k}")
    print(f"  weights: \"{best_w}\"")


if __name__ == "__main__":
    tune()