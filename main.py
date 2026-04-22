from utils import load_config, load_dataset, load_test_dataset, print_results, save_results, IMAGE_SIZE
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, RegressorMixin


# ---------- Low-level Feature Helper (nur numpy) ----------

def gradients(img):
    """Horizontale und vertikale Gradienten via zentrale Differenzen."""
    img = img.astype(np.float64)
    gx = np.zeros_like(img)
    gy = np.zeros_like(img)
    gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
    gy[1:-1, :] = img[2:, :] - img[:-2, :]
    return gx, gy


def block_mean(img, n_blocks_y, n_blocks_x):
    h, w = img.shape
    bh, bw = h // n_blocks_y, w // n_blocks_x
    trimmed = img[:n_blocks_y * bh, :n_blocks_x * bw]
    return trimmed.reshape(n_blocks_y, bh, n_blocks_x, bw).mean(axis=(1, 3))


def block_std(img, n_blocks_y, n_blocks_x):
    h, w = img.shape
    bh, bw = h // n_blocks_y, w // n_blocks_x
    trimmed = img[:n_blocks_y * bh, :n_blocks_x * bw]
    return trimmed.reshape(n_blocks_y, bh, n_blocks_x, bw).std(axis=(1, 3))


def rgb_to_gray(img_rgb):
    """Standard-Grayscale-Konvertierung (ITU-R BT.601)."""
    return 0.299 * img_rgb[..., 0] + 0.587 * img_rgb[..., 1] + 0.114 * img_rgb[..., 2]


# ---------- Feature-Extraktion ----------

def extract_gray_features(img, image_size):
    """Feature-Vektor für ein einzelnes Graustufenbild (444 Features bei 60x60)."""
    feats = []

    feats += [img.mean(), img.std(), img.min(), img.max(), np.median(img)]
    feats += np.percentile(img, [10, 25, 75, 90]).tolist()

    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    feats += row_means.tolist()
    feats += col_means.tolist()
    feats += np.diff(row_means).tolist()
    feats += np.diff(col_means).tolist()

    feats += block_mean(img, 2, 2).flatten().tolist()
    feats += block_mean(img, 4, 4).flatten().tolist()
    feats += block_mean(img, 8, 8).flatten().tolist()

    feats += block_std(img, 2, 2).flatten().tolist()
    feats += block_std(img, 4, 4).flatten().tolist()
    feats += block_std(img, 8, 8).flatten().tolist()

    gx, gy = gradients(img)
    grad_mag = np.sqrt(gx * gx + gy * gy)
    feats += [
        grad_mag.mean(), grad_mag.std(), grad_mag.max(),
        np.abs(gx).mean(), np.abs(gy).mean(),
        grad_mag[:image_size // 2].mean(),
        grad_mag[image_size // 2:].mean(),
        grad_mag[:, :image_size // 2].mean(),
        grad_mag[:, image_size // 2:].mean(),
    ]

    n_strips = 6
    strip_h = image_size // n_strips
    for i in range(n_strips):
        strip = grad_mag[i * strip_h:(i + 1) * strip_h]
        feats += [strip.mean(), strip.std()]

    hist, _ = np.histogram(img, bins=8, range=(img.min(), img.max() + 1e-9))
    hist = hist / hist.sum()
    feats += hist.tolist()

    return feats


def extract_color_features(img_rgb, image_size):
    """Kompakte Farb-Features.

    Idee: Nicht alle Features x3 pro Kanal (würde bei 3000 Samples overfitten),
    sondern gezielt Farbinfo die für Distanz / Szenentyp relevant ist:
    - Globale Mean/Std pro Kanal (6)
    - 3x3 Grid Mean pro Kanal (27) => grobe Farbverteilung
    - Obere/untere Hälfte pro Kanal (6) => Himmel vs Boden
    - Farbverhältnisse R/G, R/B, G/B (3) => Szenentyp (draussen/drinnen)
    """
    feats = []
    R, G, B = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]

    for C in (R, G, B):
        feats += [C.mean(), C.std()]

    for C in (R, G, B):
        feats += block_mean(C, 3, 3).flatten().tolist()

    half = image_size // 2
    for C in (R, G, B):
        feats += [C[:half].mean(), C[half:].mean()]

    feats += [
        R.mean() / (G.mean() + 1.0),
        R.mean() / (B.mean() + 1.0),
        G.mean() / (B.mean() + 1.0),
    ]

    return feats


def extract_features(images_flat, image_size, use_rgb=False):
    """
    use_rgb=False: (n, H*W) -> nur Grayscale-Features
    use_rgb=True:  (n, H*W*3) -> Grayscale-Features auf Luma + Color-Features
    """
    n = images_flat.shape[0]
    if use_rgb:
        images = images_flat.reshape(n, image_size, image_size, 3)
    else:
        images = images_flat.reshape(n, image_size, image_size)

    out = []
    for img in images:
        if use_rgb:
            gray = rgb_to_gray(img)
            feats = extract_gray_features(gray, image_size)
            feats += extract_color_features(img, image_size)
        else:
            feats = extract_gray_features(img, image_size)
        out.append(feats)

    return np.array(out)


# ---------- Modelle ----------

def build_hgb(params, seed):
    return HistGradientBoostingRegressor(
        max_iter=params.get("max_iter", 800),
        learning_rate=params.get("learning_rate", 0.03),
        max_depth=params.get("max_depth", 8),
        l2_regularization=params.get("l2_regularization", 0.5),
        min_samples_leaf=params.get("min_samples_leaf", 20),
        loss=params.get("loss", "absolute_error"),
        random_state=seed,
    )


def build_knn(params):
    return KNeighborsRegressor(
        n_neighbors=params.get("n_neighbors", 10),
        weights=params.get("weights", "distance"),
        n_jobs=-1,
    )


# ---------- Stacking: kNN-Prediction als zusätzliches Feature für HGB ----------

class KnnStackedHGB(BaseEstimator, RegressorMixin):
    """HGB bekommt als zusätzliches Feature die Out-of-Fold kNN-Prediction.

    Warum Out-of-Fold: Wenn kNN auf Trainingsdaten predicted, findet es sich
    selbst als nächsten Nachbar -> unrealistisch gute Prediction -> HGB lernt
    dem Signal zu stark zu vertrauen. Mit OOF bekommt HGB ein ehrliches Signal.
    """

    def __init__(self, hgb_params=None, knn_params=None, seed=42, n_folds=5):
        self.hgb_params = hgb_params or {}
        self.knn_params = knn_params or {}
        self.seed = seed
        self.n_folds = n_folds

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)
        oof_knn = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            knn_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", build_knn(self.knn_params)),
            ])
            knn_pipe.fit(X[train_idx], y[train_idx])
            oof_knn[val_idx] = knn_pipe.predict(X[val_idx])

        # kNN auf gesamten Trainingsdaten fitten (für spätere Test-Predictions)
        self.knn_full_ = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", build_knn(self.knn_params)),
        ])
        self.knn_full_.fit(X, y)

        # HGB auf Features + OOF-kNN-Prediction
        X_stacked = np.column_stack([X, oof_knn])
        self.hgb_ = build_hgb(self.hgb_params, self.seed)
        self.hgb_.fit(X_stacked, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        knn_pred = self.knn_full_.predict(X)
        X_stacked = np.column_stack([X, knn_pred])
        return self.hgb_.predict(X_stacked)


# ---------- Blend: gewichteter Mittelwert HGB + kNN ----------

class BlendedHgbKnn(BaseEstimator, RegressorMixin):
    """Trainiert HGB und kNN separat, mittelt Predictions gewichtet."""

    def __init__(self, hgb_params=None, knn_params=None, seed=42, w_hgb=0.7):
        self.hgb_params = hgb_params or {}
        self.knn_params = knn_params or {}
        self.seed = seed
        self.w_hgb = w_hgb

    def fit(self, X, y):
        self.hgb_ = build_hgb(self.hgb_params, self.seed)
        self.hgb_.fit(X, y)

        self.knn_ = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", build_knn(self.knn_params)),
        ])
        self.knn_.fit(X, y)
        return self

    def predict(self, X):
        return self.w_hgb * self.hgb_.predict(X) + (1 - self.w_hgb) * self.knn_.predict(X)


# ---------- Modell-Dispatcher ----------

def build_model(config):
    strategy = config.get("strategy", "hgb").lower()
    hgb_params = config.get("model_params", {})
    knn_params = config.get("knn_params", {})
    seed = config["random_seed"]

    if strategy == "hgb":
        return build_hgb(hgb_params, seed)
    if strategy == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", build_knn(knn_params)),
        ])
    if strategy == "stacking":
        return KnnStackedHGB(hgb_params=hgb_params, knn_params=knn_params, seed=seed)
    if strategy == "blend":
        return BlendedHgbKnn(hgb_params=hgb_params, knn_params=knn_params,
                             seed=seed, w_hgb=config.get("w_hgb", 0.7))
    raise ValueError(f"Unknown strategy: {strategy}")


# ==================== MAIN ====================

if __name__ == "__main__":
    config = load_config()

    # Daten laden
    images, distances = load_dataset(config)
    use_rgb = config.get("load_rgb", False)
    strategy = config.get("strategy", "hgb").lower()
    print(f"[INFO]: Dataset loaded with {len(images)} samples. RGB={use_rgb}, strategy='{strategy}'")

    # Feature-Extraktion
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]
    features = extract_features(images, image_size, use_rgb=use_rgb)
    print(f"[INFO]: Feature dim = {features.shape[1]}")

    # Train / Val Split
    X_train, X_val, y_train, y_val = train_test_split(
        features, distances,
        test_size=config["val_size"],
        random_state=config["random_seed"],
    )

    # Modell trainieren (Scaler ist in kNN/Stacking/Blend intern, für HGB nicht nötig)
    model = build_model(config)
    print(f"[INFO]: Training {model.__class__.__name__}")
    model.fit(X_train, y_train)

    # Auswertung
    print("\n--- Train ---")
    print_results(y_train, model.predict(X_train))
    print("\n--- Validation ---")
    print_results(y_val, model.predict(X_val))

    # Cross-Validation
    if config.get("use_cv", False):
        print(f"\n[INFO]: Running {config['cv_folds']}-fold CV for strategy='{strategy}'...")
        cv_model = build_model(config)
        kf = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_seed"])
        scores = cross_val_score(cv_model, features, distances, cv=kf,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
        mae_cm = -scores * 100
        print(f"[INFO]: CV MAE per fold (cm): {[f'{s:.2f}' for s in mae_cm]}")
        print(f"[INFO]: CV MAE mean +/- std (cm): {mae_cm.mean():.2f} +/- {mae_cm.std():.2f}")

    # Testset vorhersagen + speichern
    if config.get("save_predictions", True):
        test_images = np.array(load_test_dataset(config))
        test_features = extract_features(test_images, image_size, use_rgb=use_rgb)
        test_pred = model.predict(test_features)
        save_results(test_pred)
        print(f"\n[INFO]: Saved {len(test_pred)} predictions to prediction.csv")