from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np

from skimage.filters import sobel
from skimage.measure import block_reduce

# sklearn imports (SVRs are not allowed in this project)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor
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
    raise ValueError(f"Unknown model: {name}")


if __name__ == "__main__":
    # --- Load config ---
    config = load_config()

    # --- Load training data ---
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")
    print(f"[INFO]: Feature dim = {images.shape[1]}, label range = "
          f"[{distances.min():.2f}, {distances.max():.2f}] m")
    
    # --- Extract handcrafted features (optional) ---
    if config.get("use_handcrafted_features", False):
        print("[INFO]: Extracting handcrafted features...")
        from utils import IMAGE_SIZE
        image_size = IMAGE_SIZE[0] // config["downsample_factor"]
        images = extract_image_features(images, image_size, config["load_rgb"])
        print(f"[INFO]: New feature dim after handcrafted features = {images.shape[1]}")

    # --- Train / Validation split ---
    X_train, X_val, y_train, y_val = train_test_split(
        images, distances,
        test_size=config["val_size"],
        random_state=config["random_seed"],
    )
    print(f"[INFO]: Train = {X_train.shape[0]}, Val = {X_val.shape[0]}")

    # --- Preprocessing: scaling ---
    if config["use_scaler"]:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        print("[INFO]: Applied StandardScaler.")

    # --- Dimensionality reduction: PCA ---
    if config["use_pca"]:
        pca = PCA(n_components=config["pca_components"], random_state=config["random_seed"])
        X_train = pca.fit_transform(X_train)
        X_val = pca.transform(X_val)
        print(f"[INFO]: PCA -> {config['pca_components']} components "
              f"(explained variance: {pca.explained_variance_ratio_.sum():.3f})")

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

    # --- Predict on test set & save ---
    if config.get("save_predictions", True):
        print("\n[INFO]: Loading test set and predicting...")
        test_images = np.array(load_test_dataset(config))  # list -> np.array

        # Apply same handcrafted feature extraction as for training
        if config.get("use_handcrafted_features", False):
            print("[INFO]: Extracting handcrafted features for test set...")
            from utils import IMAGE_SIZE
            image_size = IMAGE_SIZE[0] // config["downsample_factor"]
            test_images = extract_image_features(test_images, image_size, config["load_rgb"])

        if config["use_scaler"]:
            test_images = scaler.transform(test_images)
        if config["use_pca"]:
            test_images = pca.transform(test_images)

        test_pred = model.predict(test_images)
        save_results(test_pred)
        print(f"[INFO]: Saved {len(test_pred)} predictions to prediction.csv")