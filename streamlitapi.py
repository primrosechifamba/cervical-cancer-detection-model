
import pickle
import streamlit as st
import tensorflow as tf
import keras 
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import cv2
from collections import defaultdict
from urllib import request
import os
import pandas as pd
import numpy as np
from urllib import request
import cv2
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
import os
from tensorflow.keras.applications import VGG16
import json
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder
import tensorflow
import tensorflow as tf
from collections import deque
import pandas as pd
import numpy as np
import cv2
import os
from tensorflow.keras.applications import MobileNetV2, Xception
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split

    # Replace  with your actual model file path
model = load_model('cervical_cancer_detection_model.keras', "rb")

st.title("cervical Cancer Detection Model")
    



def preprocess_image(image, target_size=(224, 224)):
    # Resize and normalize (example for a specific model)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0
    return image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg","bmp", "png"])
predict_button = st.button("Predict")  # Include the button
if predict_button:  # If the "Predict" button is clicked
    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)  # Assuming color images
        preprocessed_image = preprocess_image(image)

        predictions = model.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_class = np.argmax(predictions[0])
        predicted_proba = predictions[0][predicted_class]

        st.image(image, caption='Uploaded Image')
        st.write(f"Predicted Class: {predicted_class}")
        st.write(f"Prediction Probability: {predicted_proba:.2f}")

    
