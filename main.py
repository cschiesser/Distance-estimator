from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports...
from sklearn.model_selection import train_test_split

# SVRs are not allowed in this project.

def print_dataset_plot():
    print("Dataset ARRAY")
    print(images.shape)
    print(distances.shape)

    print(images[0].max())

    Plotting distance Lables
    X = np.arange(len(distances))   # integer indices: [0, 1, 2, ..., 2999]
    Y = distances     

    fig, ax = plt.subplots()
    ax.plot(X,Y)
    plt.show()

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

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

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




#HOW THE DATASET LOOKS LIKE
print_dataset_plot()


    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

