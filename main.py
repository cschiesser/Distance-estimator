from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import numpy as np

# sklearn imports (SVRs are not allowed in this project)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor


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

        if config["use_scaler"]:
            test_images = scaler.transform(test_images)
        if config["use_pca"]:
            test_images = pca.transform(test_images)

        test_pred = model.predict(test_images)
        save_results(test_pred)
        print(f"[INFO]: Saved {len(test_pred)} predictions to prediction.csv")