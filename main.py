from utils import load_config, load_dataset, load_test_dataset, print_results, save_results, IMAGE_SIZE
import numpy as np

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, ExtraTreesRegressor
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
    """Feature-Vektor fuer ein einzelnes Graustufenbild.

    Feature-Gruppen: Globale Stats, Row/Col-Profile, Block-Means/Stds (2/4/8),
    Gradienten-Magnitude, Gradienten-Streifen, Histogram.
    """
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

    - Globale Mean/Std pro Kanal (6)
    - 3x3 Grid Mean pro Kanal (27)
    - Obere/untere Haelfte pro Kanal (6)
    - Farbverhaeltnisse R/G, R/B, G/B (3)
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


# ---------- Modell-Bausteine ----------

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
        n_neighbors=params.get("n_neighbors", 3),
        weights=params.get("weights", "distance"),
        n_jobs=-1,
    )


def build_et(params, seed):
    """ExtraTrees als zweites Base-Modell neben kNN.

    ExtraTrees randomisiert nicht nur Feature-Subsets sondern auch Split-Werte
    -> staerker dekorreliert zu Gradient Boosting als RandomForest.
    """
    return ExtraTreesRegressor(
        n_estimators=params.get("n_estimators", 500),
        max_features=params.get("max_features", "sqrt"),
        min_samples_leaf=params.get("min_samples_leaf", 5),
        n_jobs=-1,
        random_state=seed,
    )


# ---------- Stacking: kNN + ExtraTrees als OOF-Features fuer HGB ----------

class KnnStackedHGB(BaseEstimator, RegressorMixin):
    """HGB bekommt als zusaetzliche Features die Out-of-Fold Predictions
    von kNN und ExtraTrees.

    Warum OOF: Wenn Base-Learner auf Trainingsdaten predicten wuerden,
    kennen sie die Samples bereits -> unrealistisch gute Predictions -> HGB
    ueberschaetzt das Signal. Mit OOF bekommt HGB eine ehrliche Einschaetzung,
    wie nuetzlich die Base-Learner auf neuen Daten sind.

    Warum 2 Base-Learner: kNN (instanz-basiert, lokal) und ExtraTrees
    (tree-basiert, zufaellige Splits) machen strukturell andere Fehler als
    HGB. HGB lernt welchem Signal es wann vertrauen soll.
    """

    def __init__(self, hgb_params=None, knn_params=None, et_params=None,
                 seed=42, n_folds=5):
        self.hgb_params = hgb_params or {}
        self.knn_params = knn_params or {}
        self.et_params = et_params or {}
        self.seed = seed
        self.n_folds = n_folds

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # OOF-Predictions: zwei Spalten, eine pro Base-Learner
        oof_knn = np.zeros(len(y))
        oof_et = np.zeros(len(y))

        for train_idx, val_idx in kf.split(X):
            # kNN-Fold
            knn_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("knn", build_knn(self.knn_params)),
            ])
            knn_pipe.fit(X[train_idx], y[train_idx])
            oof_knn[val_idx] = knn_pipe.predict(X[val_idx])

            # ExtraTrees-Fold (kein Scaler noetig, Tree-Modell)
            et = build_et(self.et_params, self.seed)
            et.fit(X[train_idx], y[train_idx])
            oof_et[val_idx] = et.predict(X[val_idx])

        # Base-Learner auf gesamtem Trainingsset fitten (fuer Test-Predictions)
        self.knn_full_ = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", build_knn(self.knn_params)),
        ])
        self.knn_full_.fit(X, y)

        self.et_full_ = build_et(self.et_params, self.seed)
        self.et_full_.fit(X, y)

        # HGB auf: [Original-Features, OOF-kNN, OOF-ET]
        X_stacked = np.column_stack([X, oof_knn, oof_et])
        self.hgb_ = build_hgb(self.hgb_params, self.seed)
        self.hgb_.fit(X_stacked, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        knn_pred = self.knn_full_.predict(X)
        et_pred = self.et_full_.predict(X)
        X_stacked = np.column_stack([X, knn_pred, et_pred])
        return self.hgb_.predict(X_stacked)


# ---------- Multi-Seed Bagging Wrapper ----------

class BaggedStackedHGB(BaseEstimator, RegressorMixin):
    """Wrapper der n KnnStackedHGB-Modelle mit verschiedenen Seeds trainiert
    und die Predictions mittelt.

    Warum: Jeder Seed produziert leicht andere Folds und andere Baum-Strukturen
    in ExtraTrees und HGB. Die systematischen Fehler (Bias) bleiben, die
    zufaelligen (Varianz) mitteln sich raus -> stabileres, besseres Modell.

    Kosten: n-fache Trainingszeit. Inference ist auch n-mal, aber immer noch
    schnell genug fuer 622 Test-Samples.
    """

    def __init__(self, hgb_params=None, knn_params=None, et_params=None,
                 seeds=(42, 123, 456, 789, 1000), n_folds=5):
        self.hgb_params = hgb_params or {}
        self.knn_params = knn_params or {}
        self.et_params = et_params or {}
        self.seeds = tuple(seeds)
        self.n_folds = n_folds

    def fit(self, X, y):
        self.models_ = []
        for seed in self.seeds:
            m = KnnStackedHGB(
                hgb_params=self.hgb_params,
                knn_params=self.knn_params,
                et_params=self.et_params,
                seed=seed,
                n_folds=self.n_folds,
            )
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return preds.mean(axis=1)


def build_model(config):
    """Multi-Seed Bagging ueber kNN+ExtraTrees Stacking mit HGB als Meta-Learner."""
    return BaggedStackedHGB(
        hgb_params=config.get("model_params", {}),
        knn_params=config.get("knn_params", {}),
        et_params=config.get("et_params", {}),
        seeds=config.get("bagging_seeds", [42, 123, 456, 789, 1000]),
    )


# ==================== MAIN ====================

if __name__ == "__main__":
    config = load_config()

    # Daten laden
    images, distances = load_dataset(config)
    use_rgb = config.get("load_rgb", False)
    print(f"[INFO]: Dataset loaded with {len(images)} samples. RGB={use_rgb}")

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

    # Modell trainieren
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
        print(f"\n[INFO]: Running {config['cv_folds']}-fold CV...")
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