from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np

from skimage.filters import sobel
from skimage.measure import block_reduce
from skimage.feature import hog

# sklearn imports (SVRs are not allowed in this project)
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor

def extract_image_features(images_flat, image_size, is_rgb):
    """
    Berechnet handcrafted Features aus den Bildern.
    
    Args:
        images_flat: np.array (N, features_flat) - geflattened Bilder
        image_size: int - Höhe/Breite (z.B. 60 für 60x60)
        is_rgb: bool - True wenn RGB, False wenn grayscale
    
    Returns:
        np.array (N, num_features)
    """
    n_samples = images_flat.shape[0]
    channels = 3 if is_rgb else 1
    
    # Bilder zurück in 2D (oder 3D für RGB) umformen
    images = images_flat.reshape(n_samples, image_size, image_size, channels)
    
    feature_list = []
    
    for i in range(n_samples):
        img = images[i]  # shape: (H, W, C)
        
        # Falls RGB → in grayscale konvertieren für Edge-Features
        if is_rgb:
            gray = img.mean(axis=-1)  # einfacher Mittelwert über Kanäle
        else:
            gray = img[..., 0]  # einziger Kanal
        
        features = []
        
        # 1. Globale Statistiken (5 Features pro Kanal)
        for c in range(channels):
            ch = img[..., c]
            features.extend([
                ch.mean(), ch.std(), ch.min(), ch.max(), np.median(ch)
            ])
        
        # 2. Helligkeit pro Zeile (H Features) - nur grayscale
        row_means = gray.mean(axis=1)  # shape: (H,)
        features.extend(row_means.tolist())
        
        # 3. Helligkeit pro Spalte (W Features) - nur grayscale
        col_means = gray.mean(axis=0)  # shape: (W,)
        features.extend(col_means.tolist())
        
        # 4. 3x3 Grid Mittelwerte (9 Features)
        grid_size = image_size // 3
        grid_features = block_reduce(gray, block_size=(grid_size, grid_size), func=np.mean)
        features.extend(grid_features.flatten()[:9].tolist())
        
        # 5. Edge-Features (Sobel) - 5 Features
        edges = sobel(gray)
        features.extend([
            edges.mean(), edges.std(), edges.max(),
            edges[image_size//2:].mean(),  # untere Bildhälfte
            edges[:image_size//2].mean(),  # obere Bildhälfte
        ])

        # 6. HOG-Features (Histogram of Oriented Gradients)
        # Beschreibt lokale Kantenrichtungen - sehr stark für Objekterkennung
        hog_feats = hog(
            gray,
            orientations=8,           # 8 Richtungs-Bins (horizontal, diagonal, ...)
            pixels_per_cell=(10, 10), # bei 60x60 Bild: 6x6 = 36 Zellen
            cells_per_block=(2, 2),   # je 2x2 Zellen werden zusammen normalisiert
            feature_vector=True,
        )
        features.extend(hog_feats.tolist())
        
        feature_list.append(features)
    
    return np.array(feature_list)

def build_model(name: str, params: dict, random_seed: int):
    """Factory: pick a regressor based on the config."""
    name = name.lower()
    if name == "ridge":
        return Ridge(alpha=params.get("alpha", 1.0), random_state=random_seed)
    if name == "lasso":
        return Lasso(alpha=params.get("alpha", 0.001), random_state=random_seed, max_iter=10000)
    if name == "knn":
        return KNeighborsRegressor(
            n_neighbors=params.get("n_neighbors", 5),
            weights=params.get("weights", "distance"),
        )
    if name == "rf":
        return RandomForestRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", None),
            n_jobs=-1,
            random_state=random_seed,
        )
    if name == "gbr":
        return GradientBoostingRegressor(
            n_estimators=params.get("n_estimators", 200),
            max_depth=params.get("max_depth", 3),
            learning_rate=params.get("learning_rate", 0.1),
            random_state=random_seed,
        )
    if name == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=params.get("max_iter", 500),
            max_depth=params.get("max_depth", None),
            learning_rate=params.get("learning_rate", 0.05),
            l2_regularization=params.get("l2_regularization", 0.0),
            random_state=random_seed,
        )
    if name == "mlp":
        return MLPRegressor(
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", [128, 64])),
            max_iter=params.get("max_iter", 500),
            random_state=random_seed,
        )
    if name == "ensemble":
        # Ensemble: HGB + RF (gemittelt)
        hgb = HistGradientBoostingRegressor(
            max_iter=params.get("hgb_max_iter", 600),
            learning_rate=params.get("hgb_learning_rate", 0.05),
            max_depth=params.get("hgb_max_depth", 6),
            l2_regularization=params.get("hgb_l2", 0.3),
            random_state=random_seed,
        )
        rf = RandomForestRegressor(
            n_estimators=params.get("rf_n_estimators", 300),
            max_depth=params.get("rf_max_depth", 20),
            n_jobs=-1,
            random_state=random_seed,
        )
        return VotingRegressor(estimators=[("hgb", hgb), ("rf", rf)], n_jobs=-1)
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    # --- Load config ---
    config = load_config()

    # --- Load training data ---
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    print(f"[INFO]: Feature dim = {images.shape[1]}, label range = "
          f"[{distances.min():.2f}, {distances.max():.2f}] m")
    
    # --- Feature engineering: handcrafted, raw pixels, or both ---
    from utils import IMAGE_SIZE
    image_size = IMAGE_SIZE[0] // config["downsample_factor"]

    pixel_features = images  # original flat pixel features
    handcrafted_features = None

    if config.get("use_handcrafted_features", False):
        print("[INFO]: Extracting handcrafted features...")
        handcrafted_features = extract_image_features(images, image_size, config["load_rgb"])
        print(f"[INFO]: Handcrafted feature dim = {handcrafted_features.shape[1]}")

    # --- Train / Validation split (on pixel features) ---
    indices = np.arange(len(distances))
    X_train_pix, X_val_pix, y_train, y_val, idx_train, idx_val = train_test_split(
        pixel_features, distances, indices,
        test_size=config["val_size"],
        random_state=config["random_seed"],
    )
    print(f"[INFO]: Train = {X_train_pix.shape[0]}, Val = {X_val_pix.shape[0]}")

    # --- Preprocessing: scaling on pixels ---
    if config["use_scaler"]:
        scaler = StandardScaler()
        X_train_pix = scaler.fit_transform(X_train_pix)
        X_val_pix = scaler.transform(X_val_pix)
        print("[INFO]: Applied StandardScaler on pixels.")

    # --- Dimensionality reduction: PCA on pixels only ---
    if config["use_pca"]:
        pca = PCA(n_components=config["pca_components"], random_state=config["random_seed"])
        X_train_pix = pca.fit_transform(X_train_pix)
        X_val_pix = pca.transform(X_val_pix)
        print(f"[INFO]: PCA -> {config['pca_components']} components "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

    # --- Combine with handcrafted features (if present) ---
    if handcrafted_features is not None and config.get("combine_with_pixels", False):
        # Scale handcrafted separately so they're on similar scale as PCA features
        hc_scaler = StandardScaler()
        hc_train = hc_scaler.fit_transform(handcrafted_features[idx_train])
        hc_val = hc_scaler.transform(handcrafted_features[idx_val])
        X_train = np.hstack([hc_train, X_train_pix])
        X_val = np.hstack([hc_val, X_val_pix])
        print(f"[INFO]: Combined features: {X_train.shape[1]} (handcrafted + PCA pixels)")
    elif handcrafted_features is not None:
        # Only handcrafted, no pixels
        hc_scaler = StandardScaler()
        X_train = hc_scaler.fit_transform(handcrafted_features[idx_train])
        X_val = hc_scaler.transform(handcrafted_features[idx_val])
        print(f"[INFO]: Using only handcrafted features: {X_train.shape[1]}")
    else:
        # Only PCA pixels
        X_train, X_val = X_train_pix, X_val_pix

    # --- Train model ---
    model = build_model(config["model"], config.get("model_params", {}), config["random_seed"])
    print(f"[INFO]: Training model: {model.__class__.__name__}")
    model.fit(X_train, y_train)

    # --- Evaluate on train and validation ---
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    print("\n--- Train ---")
    print_results(y_train, train_pred)
    print("\n--- Validation ---")
    print_results(y_val, val_pred)

    # --- Optional: Grid Search for hyperparameter tuning ---
    if config.get("use_grid_search", False):
        print("\n[INFO]: Running GridSearchCV - this will take a while...")
        
        # Grid für HGB
        param_grid = {
            "max_iter": [400, 600, 800],
            "learning_rate": [0.03, 0.05, 0.08],
            "max_depth": [4, 6, 8],
            "l2_regularization": [0.1, 0.5],
        }
        
        base_model = HistGradientBoostingRegressor(random_state=config["random_seed"])
        
        # Use the SAME features as for training
        if handcrafted_features is not None and not config.get("combine_with_pixels", False):
            gs_features = handcrafted_features
        else:
            gs_features = pixel_features
        
        gs_scaler = StandardScaler()
        gs_features_scaled = gs_scaler.fit_transform(gs_features)
        
        kf_gs = KFold(n_splits=3, shuffle=True, random_state=config["random_seed"])  # 3-fold for speed
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=kf_gs,
            scoring="neg_mean_absolute_error",
            n_jobs=-1,
            verbose=2,
        )
        grid_search.fit(gs_features_scaled, distances)
        
        print(f"\n[INFO]: BEST PARAMS: {grid_search.best_params_}")
        print(f"[INFO]: BEST CV MAE (cm): {-grid_search.best_score_ * 100:.2f}")

    # --- Optional: Cross-Validation for honest performance estimate ---
    if config.get("use_cv", False):
        print(f"\n[INFO]: Running {config['cv_folds']}-fold cross-validation...")
        # Build the same pipeline manually for CV
        from sklearn.pipeline import Pipeline
        cv_steps = []
        if config["use_scaler"]:
            cv_steps.append(("scaler", StandardScaler()))
        if config["use_pca"]:
            cv_steps.append(("pca", PCA(n_components=config["pca_components"], random_state=config["random_seed"])))
        cv_steps.append(("model", build_model(config["model"], config.get("model_params", {}), config["random_seed"])))
        cv_pipe = Pipeline(cv_steps)

        # Use the SAME features as for training (handcrafted or pixels)
        if handcrafted_features is not None and not config.get("combine_with_pixels", False):
            cv_features = handcrafted_features
        else:
            cv_features = pixel_features

        kf = KFold(n_splits=config["cv_folds"], shuffle=True, random_state=config["random_seed"])
        scores = cross_val_score(cv_pipe, cv_features, distances, cv=kf,
                                 scoring="neg_mean_absolute_error", n_jobs=-1)
        mae_cm = -scores * 100
        print(f"[INFO]: CV MAE per fold (cm): {[f'{s:.2f}' for s in mae_cm]}")
        print(f"[INFO]: CV MAE mean ± std (cm): {mae_cm.mean():.2f} ± {mae_cm.std():.2f}")

    # --- Predict on test set & save ---
    if config.get("save_predictions", True):
        print("\n[INFO]: Loading test set and predicting...")
        test_pixels = np.array(load_test_dataset(config))

        # Optional: extract handcrafted features for test set
        test_hc = None
        if config.get("use_handcrafted_features", False):
            print("[INFO]: Extracting handcrafted features for test set...")
            test_hc = extract_image_features(test_pixels, image_size, config["load_rgb"])

        # Apply scaler + PCA on pixels
        if config["use_scaler"]:
            test_pixels = scaler.transform(test_pixels)
        if config["use_pca"]:
            test_pixels = pca.transform(test_pixels)

        # Combine
        if test_hc is not None and config.get("combine_with_pixels", False):
            test_hc = hc_scaler.transform(test_hc)
            test_features = np.hstack([test_hc, test_pixels])
        elif test_hc is not None:
            test_features = hc_scaler.transform(test_hc)
        else:
            test_features = test_pixels

        print(f"[INFO]: Test feature dim = {test_features.shape[1]}")
        test_pred = model.predict(test_features)
        save_results(test_pred)
        print(f"[INFO]: Saved {len(test_pred)} predictions to prediction.csv")