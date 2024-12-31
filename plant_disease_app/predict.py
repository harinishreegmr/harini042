import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load disease details CSV
disease_details_df = pd.read_csv('disease_details.csv')
disease_details_df.columns = disease_details_df.columns.str.strip()

# Load the trained model
model = tf.keras.models.load_model('crop_disease_model.h5')

def preprocess_image(img_path):
    """Preprocess image for the model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict_disease(img_path):
    """Predict the disease and fetch details."""
    img_array = preprocess_image(img_path)
    predicted_class_idx = model.predict(img_array).argmax()
    details = disease_details_df.iloc[predicted_class_idx]
    return {
        "disease_name": details['disease_name'],
        "cause": details['cause'],
        "effect": details['effect'],
        "prevention": details['prevention'],
        "treatment": details['treatment']
    }
