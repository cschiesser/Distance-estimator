import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

from utils import load_config, load_dataset, load_test_dataset, save_results


def extract_region_features(flat_images, config):
    """Statistiken pro Bildregion (grid_size x grid_size)."""
    n_samples = flat_images.shape[0]
    grid = config["feature_engineering"]["grid_size"]
    img_size = 300 // config["downsample_factor"]
    n_channels = 3 if config["load_rgb"] else 1

    if config["load_rgb"]:
        images = flat_images.reshape(n_samples, img_size, img_size, n_channels)
    else:
        images = flat_images.reshape(n_samples, img_size, img_size)

    region_size = img_size // grid
    features = []

    for i in range(n_samples):
        img = images[i]
        per_image = []

        for row in range(grid):
            for col in range(grid):
                r_start = row * region_size
                r_end = (row + 1) * region_size
                c_start = col * region_size
                c_end = (col + 1) * region_size
                region = img[r_start:r_end, c_start:c_end]

                if config["load_rgb"]:
                    for ch in range(n_channels):
                        per_image.append(region[:, :, ch].mean())
                        per_image.append(region[:, :, ch].std())
                else:
                    per_image.append(region.mean())
                    per_image.append(region.std())

        per_image.append(img.mean())
        per_image.append(img.std())
        per_image.append(img.min())
        per_image.append(img.max())

        features.append(per_image)

    return np.array(features)


def build_knn(config):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=config["pca"]["n_components"],
                    random_state=config["random_state"])),
        ("knn", KNeighborsRegressor(
            n_neighbors=config["knn"]["n_neighbors"],
            weights=config["knn"]["weights"],
            n_jobs=-1,
        )),
    ])


def build_et(config):
    return ExtraTreesRegressor(
        n_estimators=config["extratrees"]["n_estimators"],
        max_depth=config["extratrees"]["max_depth"],
        min_samples_leaf=config["extratrees"]["min_samples_leaf"],
        max_features=config["extratrees"]["max_features"],
        random_state=config["random_state"],
        n_jobs=-1,
    )


def main():
    # --- Config laden ---
    config = load_config()

    # --- Trainingsdaten laden ---
    images, distances = load_dataset(config, split="train")
    print(f"Raw images shape: {images.shape}")

    # --- Feature Engineering ---
    print("Extracting region features...")
    eng_features = extract_region_features(images, config)
    print(f"Engineered features shape: {eng_features.shape}")

    combined = np.hstack([images, eng_features])
    print(f"Combined features shape: {combined.shape}")

    # --- Val-Check ---
    X_train, X_val, y_train, y_val = train_test_split(
        combined, distances,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    print("\n[Val-Check] Training KNN...")
    knn = build_knn(config)
    knn.fit(X_train, y_train)
    knn_val = knn.predict(X_val)

    print("[Val-Check] Training ExtraTrees...")
    et = build_et(config)
    et.fit(X_train, y_train)
    et_val = et.predict(X_val)

    ens_val = (knn_val + et_val) / 2
    val_mae = mean_absolute_error(y_val, ens_val) * 100
    val_r2 = r2_score(y_val, ens_val) * 100
    print(f"\nKNN:      {mean_absolute_error(y_val, knn_val) * 100:.2f} cm")
    print(f"ET:       {mean_absolute_error(y_val, et_val) * 100:.2f} cm")
    print(f"Ensemble: {val_mae:.2f} cm  (R² = {val_r2:.2f})")

    # --- Finale Modelle auf ALLEN Trainingsdaten ---
    print("\nTraining final models on full train set...")
    knn_final = build_knn(config)
    knn_final.fit(combined, distances)

    et_final = build_et(config)
    et_final.fit(combined, distances)

    # --- Testdaten laden ---
    print("Loading test set...")
    test_images = np.array(load_test_dataset(config))
    print(f"Raw test shape: {test_images.shape}")

    # Gleiche Features für Test extrahieren
    print("Extracting test features...")
    test_eng = extract_region_features(test_images, config)
    test_combined = np.hstack([test_images, test_eng])
    print(f"Combined test shape: {test_combined.shape}")

    # Vorhersagen
    knn_test = knn_final.predict(test_combined)
    et_test = et_final.predict(test_combined)
    test_pred = (knn_test + et_test) / 2

    print(f"\nPredictions: min={test_pred.min():.2f}, "
          f"max={test_pred.max():.2f}, mean={test_pred.mean():.2f}")

    # --- CSV speichern ---
    save_results(test_pred)
    print("\n✓ prediction.csv gespeichert!")


if __name__ == "__main__":
    main()