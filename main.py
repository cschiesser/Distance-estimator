from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports...
from sklearn.model_selection import train_test_split

# SVRs are not allowed in this project.
#WICHTIG ist für Erklärungsauftrag

def print_dataset_plot(bin_width=0.01, output_path="distance_distribution.png"):
    print("Dataset ARRAY")
    print(images.shape)
    print(distances.shape)

    print(images[0].max())

    # Plot a histogram: x-axis is distance, y-axis is number of samples.
    distance_values = np.asarray(distances, dtype=float)
    min_distance = distance_values.min()
    max_distance = distance_values.max()
    bins = np.arange(min_distance, max_distance + bin_width, bin_width)

    fig, ax = plt.subplots()
    ax.hist(distance_values, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")
    ax.set_title(f"Distance Distribution (bin width = {bin_width:.3f} m)")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]: Saved plot to {output_path}")


def print_feature_distribution_plot(feature_matrix, bin_width=0.05, name="x_train_scaled_distribution", output_path=None):
    # Flatten all feature values to inspect their global distribution.
    feature_values = np.asarray(feature_matrix, dtype=float).ravel()
    min_value = feature_values.min()
    max_value = feature_values.max()
    bins = np.arange(min_value, max_value + bin_width, bin_width)

    if output_path is None:
        output_path = f"{name}.png"

    fig, ax = plt.subplots()
    ax.hist(feature_values, bins=bins, edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Feature value")
    ax.set_ylabel("Count")
    ax.set_title(f"{name} (bin width = {bin_width:.3f})")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO]: Saved plot to {output_path}")

#region Project Description
"""
This model takes as input an image captured by ANYmal's camera and outputs
an estimate of the distance to the closest obstacle in the image.

Data overview:
- Features: train_images
- Labels: train_labels (distances in meters)

Image representation:
- Pixel intensity values (RGB)
- Typical image shape: W x H x 3

Processing idea:
- Flatten each image into a 1D feature vector
- Then each color component of each pixel is treated as one feature

Example:
- For an RGB image of 30 x 30 pixels: 30 x 30 x 3 = 2700 features
- Max pixel brightness value: 255
"""
#endregion

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")



    # TODO: Your implementation starts here
    
    X = images
    Y = distances

    #HOW THE DATASET LOOKS LIKE
    print_dataset_plot()

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

    #region Preprocessing Data
    """
    If you examine the images, you will notice that illumination conditions and objects in the scene vary significantly between frames. This results in variations in pixel values.

    For machine learning models to perform well, it is often important that features are scaled to a common range. Therefore, you may need to apply a scaling method from the scikit-learn preprocessing module.

    WICHTIG ->              Wir sehen im plot der raw Data, dass es outliers gibt, welche gegen oben oder unten (sehr weit oder sehr nah) sind. def print_dataset_plot()

                            scikit-learn.org dagt: "If some outliers are present in the set, robust scalers or other transformers can be more appropriate. The behaviors of the different scalers, transformers,
                            and normalizers on a dataset containing marginal outliers are highlighted in Compare the effect of different scalers on data with outliers." https://scikit-learn.org/stable/modules/preprocessing.html (7.3)

                            Für die erklärung unserer Preprocessing METHODE erwähne Plotting modelle und entscheidungen von https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#compare-the-effect-of-different-scalers-on-data-with-outliers

                            Wichtige erkennungen: in unserem raw plot sehen wir , dass es sehr wenige outliers gibt! Die vorherige seite sagt: QuantileTransformer provides non-linear transformations in which distances between marginal outliers and inliers are shrunk.
                            -> In meinen Augen sind diese outliers genau solche marginals, welche unsere unser model nicht beeinflussen obwohl sie es sollen !
                            ACTUALLY ich habe es verwechselt warte.

                            Power transform sieht besser auch, sie sagen:   Power transforms are a family of parametric, monotonic transformations that are applied to make data more Gaussian-like.
                                                                            This is useful for modeling issues related to heteroscedasticity (non-constant variance), or other situations where normality is desired.
                                                                            https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PowerTransformer.html#powertransformer

                                                                            PowerTransformer

                                                                            Parameters:     method : {‘yeo-johnson’, ‘box-cox’}, default=’yeo-johnson’
                                                                                                    The power transform method. Available methods are:  ‘yeo-johnson’ [1], works with positive and negative values
                                                                                                                                                        ‘yeo-johnson’ [1], works with positive and negative values

                                                                                            standardize :  bool, default=True
                                                                                                            Set to True to apply zero-mean, unit-variance normalization to the transformed output.

                                                                                            copy : bool, default=True
                                                                                        

                                                                            Attributes:     lambdas = ndarray of float of shape (n_features,)
                                                                                                The parameters of the power transformation for selected features

                                                                                            n_features_in : int
                                                                                                Number of features seen during fit.
                                                                                            
                                                                                            feature_names_in_ : ndarray of shape (n_features_in,)
                                                                                                Names of features seen during fit. Defined only when X has feature names that are all strings.

                            Standardization, or mean removal and variance scaling: https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling

                                    Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn;
                                    they might behave badly if the individual features do not more or less look like standard normally distributed data: Gaussian with zero mean and unit variance.

                                    LOOK AT PLOT DISTRIBUTION! 

                                    The preprocessing module provides the StandardScaler utility class, which is a quick and easy way to perform the following operation on an array-like dataset.

    """

    from sklearn import preprocessing
    
    #region Achtung: Both StandardScaler and MinMaxScaler are very sensitive to the presence of outliers. https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html#sphx-glr-auto-examples-preprocessing-plot-all-scaling-py
    scaler_bad = preprocessing.StandardScaler().fit(X_train)
    X_scaled_bad = scaler_bad.transform(X_train)

    #SO SIEHTS NEU AUS aber schlecht wegen outliers bei standartscaler
    print_feature_distribution_plot(
        X_scaled_bad,
        bin_width=0.05,
        name="x_train_standard_scaled_distribution",
    )



    #Besser -> RobustScaler 
    #Unlike the previous scalers, the centering and scaling statistics of RobustScaler are based on percentiles and are therefore not influenced by a small number of very large marginal outliers.
    from sklearn.preprocessing import RobustScaler
    robust_scaler = RobustScaler().fit(X_train)
    X_scaled_robust = robust_scaler.transform(X_train)
    print_feature_distribution_plot(
        X_scaled_robust,
        bin_width=0.05,
        name="x_train_robust_scaled_distribution",
    )


    #endregion
    #endregion
    

    from sklearn.preprocessing import QuantileTransformer

    qt = QuantileTransformer()
    X_Quantile= qt.fit_transform(X_train)
    print_feature_distribution_plot(
        X_Quantile,
        bin_width=0.05,
        name="x_train_Quantile"
    )




    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)



#region DON'T REMOVE -> GRADING SIMULATION!!!!!!###
    
    from sklearn.metrics import mean_absolute_error
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)

    # Save Kaggle submission using the test split
    test_images = load_test_dataset(config)
    test_pred = model.predict(test_images)
    save_results(test_pred)

#endregion



    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

