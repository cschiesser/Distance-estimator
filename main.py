from utils import load_config, load_dataset, load_test_dataset, print_results, save_results, IMAGE_SIZE
import numpy as np

from skimage.filters import sobel
from skimage.measure import block_reduce
from skimage.feature import hog

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor


def extract_features(images_flat, image_size):
    """Berechnet handcrafted Features aus den geflattenen grayscale Bildern."""
    n = images_flat.shape[0]
    images = images_flat.reshape(n, image_size, image_size)

    out = []
    for img in images:
        feats = []

        # Globale Statistiken
        feats += [img.mean(), img.std(), img.min(), img.max(), np.median(img)]

        # Helligkeit pro Zeile und pro Spalte
        feats += img.mean(axis=1).tolist()
        feats += img.mean(axis=0).tolist()

        # 3x3 Grid Mittelwerte
        grid = block_reduce(img, block_size=(image_size // 3, image_size // 3), func=np.mean)
        feats += grid.flatten()[:9].tolist()

        # Edges (Sobel) - global + obere/untere Hälfte
        edges = sobel(img)
        feats += [
            edges.mean(), edges.std(), edges.max(),
            edges[image_size // 2:].mean(),
            edges[:image_size // 2].mean(),
        ]

        # HOG - lokale Kantenrichtungen
        hog_feats = hog(img, orientations=8, pixels_per_cell=(10, 10),
                        cells_per_block=(2, 2), feature_vector=True)
        feats += hog_feats.tolist()

        out.append(feats)

    return np.array(out)


def build_model(name, params, seed):
    name = name.lower()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=seed)
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 300),
            max_depth=params.get("max_depth", 20),
            n_jobs=-1, random_state=seed,
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 800),
            learning_rate=params.get("learning_rate", 0.03),
            max_depth=params.get("max_depth", 8),
            l2_regularization=params.get("l2_regularization", 0.5),
            random_state=seed,
        )
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    config = load_config()

    # Daten laden
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")

    # Feature Extraction
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]
    features = extract_features(images, image_size)
    print(f"[INFO]: Feature dim = {features.shape[1]}")

    # Train / Validation Split
    X_train, X_val, y_train, y_val = train_test_split(
        features, distances,
        test_size=config["val_size"],
        random_state=config["random_seed"],
    )

    # Skalieren
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    # Modell trainieren
    model = build_model(config["model"], config.get("model_params", {}), config["random_seed"])
    print(f"[INFO]: Training {model.__class__.__name__}")
    model.fit(X_train, y_train)

    # Auswertung Train + Val
    print("\n--- Train ---")
    print_results(y_train, model.predict(X_train))
    print("\n--- Validation ---")
    print_results(y_val, model.predict(X_val))

    # Cross-Validation für ehrlichere Performance-Schätzung
    if config.get("use_cv", False):
        print(f"\n[INFO]: Running {config['cv_folds']}-fold cross-validation...")
        features_scaled = scaler.fit_transform(features)
        cv_model = build_model(config["model"], config.get("model_params", {}), config["random_seed"])
        kf = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_seed"])
        scores = cross_val_score(cv_model, features_scaled, distances, cv=kf,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
        mae_cm = -scores * 100
        print(f"[INFO]: CV MAE per fold (cm): {[f'{s:.2f}' for s in mae_cm]}")
        print(f"[INFO]: CV MAE mean ± std (cm): {mae_cm.mean():.2f} ± {mae_cm.std():.2f}")

    # Testset vorhersagen + speichern
    if config.get("save_predictions", True):
        test_images = np.array(load_test_dataset(config))
        test_features = extract_features(test_images, image_size)
        test_features = scaler.transform(test_features)
        test_pred = model.predict(test_features)
        save_results(test_pred)
        print(f"\n[INFO]: Saved {len(test_pred)} predictions to prediction.csv")