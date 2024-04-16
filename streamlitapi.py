
import streamlit as st
import tensorflow as tf
import keras 
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Dropout,Dense
from keras.models import Model
from keras.callbacks import EarlyStopping,ModelCheckpoint
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

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

    
