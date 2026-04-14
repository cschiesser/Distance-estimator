from utils import load_config, load_dataset, load_test_dataset, print_results, save_results
import matplotlib.pyplot as plt
import numpy as np

# sklearn imports...
from sklearn.model_selection import train_test_split

# SVRs are not allowed in this project.

    ###It takes as input an image captured by ANYmal's camera and it outputs an estimate of the distance to the closest obstacle in the image.

    #What we have

    #train_images : Here you see the robot's observations from its camera while it is travelling around an office.

    #train_labels : Here you see the distance to the closest obstacle for each image in meters. 

    

    #FEATURES -> train_images
    #LABLES -> train_lables (distances)

    #Images are represented by **pixel intensity values** 
    #In a typical COLOUR image, pixel colours are obtained by mixing the primary colours red, green, and blue (RGB).

    #Therefore, an image can be described as a matrix (M) of dimensions:        **W x H x3**

    #PROCESSING

    #To treat images as feature vectors, we FLATTEN the images, i.e. the rows of the image matrices are concatenated
    #sequentially in a single row. In a colour image, we further flatten the colour dimension similarly.


    #Then, each colour component of each pixel value can be thought of as a feature.
    #For example, an RGB image with height = 30 pixels and width = 30 pixels gives us ** 30 x 30 x 3 = 2700 **
    #Max brightness = 255

if __name__ == "__main__":
    # Load configs from "config.yaml"
    config = load_config()

    # Load dataset: images and corresponding minimum distance values
    images, distances = load_dataset(config)
    print(f"[INFO]: Dataset loaded with {len(images)} samples.")



    # TODO: Your implementation starts here
    
    X = images
    Y = distances

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(X_train,y_train)



    ###DON'T REMOVE -> GRADING SIMULATION!!!!!!###
    from sklearn.metrics import mean_absolute_error
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    print(mae)
    ###DON'T REMOVE -> GRADING SIMULATION!!!!!!###




    ##HOW THE DATASET LOOKS LIKE

    #print("Dataset ARRAY")
    #print(images.shape)
    #print(distances.shape)

    #print(images[0].max())

    #Plotting distance Lables
    #X = np.arange(len(distances))   # integer indices: [0, 1, 2, ..., 2999]
    #Y = distances     

    #fig, ax = plt.subplots()
    #ax.plot(X,Y)
    #plt.show()



    # possible preprocessing steps ... training the model

    # Evaluation
    # print_results(gt, pred)

    # Save the results
    # save_results(test_pred)