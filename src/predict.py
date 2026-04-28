import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

IMG_SIZE = (224, 224)

def load_model(model_path):
    return tf.keras.models.load_model(model_path)

def predict_height(model, img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return float(prediction[0][0])