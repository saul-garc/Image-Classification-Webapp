import os
import streamlit as st
import pathlib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import matplotlib.image as mpimg
import random
from PIL import Image, ImageOps

st.title("Image Classification Webapp")
st.markdown("---")
#st.subheader("This simple webapp demonstrates image classification")
st.text("try it out below:")

# TAKING USER INPUT 
uploaded_file = st.file_uploader("Upload picture", type=['jpeg', 'jpg', 'png'], accept_multiple_files=False)  

# LOAD MODEL
model = tf.keras.models.load_model('best_model.keras')

# GET CLASSNAMES FROM DATASET
data_dir = pathlib.Path("mini-dataset/classes/")  
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # make list of class names from subdirectories


# VIEWING RANDOM IMAGES FROM DATASET
def get_random_sample_images():
    """
    Function for viewing random images in mini dataset directory.
    Returns list of image for each class
    """
    test_image_samples = []

    for img_class in class_names:
        img_dir = "mini-dataset/classes/" + img_class + "/"
        random_image = random.sample(os.listdir(img_dir), 1)
        img = mpimg.imread(img_dir + "/" + random_image[0])
        test_image_samples.append(img)

    return test_image_samples


# PREPROCESSING USER INPUT AND PREDICTING 
def import_and_predict(image_data, model=model):
    """
    Reads image from user input, turns it into numpy array,
    reshapes it to expected input (1,150,150, 3), and normalizes it.
    """
    size = (150,150)
    img = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(img)
    img = img[np.newaxis,...]
    img=img/255.0

    pred = model.predict(img)
    return pred


def main():
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=350)

        prediction = import_and_predict(image)
        print(prediction)
        pred_class = class_names[prediction.argmax()]
        st.write("### Prediction:", "`",pred_class,"`")

    st.write("---")
    st.subheader("Sample images of each class contained in dataset:")

    sample_images = get_random_sample_images()
    st.image(sample_images, width=234, caption=class_names)
    

if __name__ == "__main__":
    main()