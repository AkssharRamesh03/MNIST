from flask import Blueprint, request, render_template
from keras.models import load_model
import skimage.io as io
import tensorflow as tf
import numpy as np
from PIL import Image
import io

main = Blueprint('main', __name__)


@main.route('/')
def index():
    return "image classification model"


def normalize(image):
    images = tf.cast(image, tf.float32)
    images = np.array(images).reshape(1, 28, 28, 1)
    images /= 255
    return images


@main.route('/predict', methods=['POST'])
def predict():
    # labels
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress',
                   'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # Get the image file from the POST request
    file = request.files['image']
    image = Image.open(io.BytesIO(file.read()))
    # Convert the image to a grayscale image
    gray_image = image.convert('L')
    image_resized = gray_image.resize((28, 28))
    # Normalize the image array
    image_normalized = normalize(image_resized)
    # Load the model

    model = load_model('model/mnist_classifier_model.h5')
    prediction = model.predict(image_normalized)
    return class_names[np.argmax(prediction)]
