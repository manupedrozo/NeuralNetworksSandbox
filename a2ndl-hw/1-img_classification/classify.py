import os
import numpy as np
from PIL import Image
from enum import Enum

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

from utils import create_csv, get_files_in_directory

from classes import classes

"""
Generate csv for submission with the last model trained for the chosen architecture (MODEL_CHOICE).
"""

class Model(Enum):
    VGG = 1
    HOMEBREW = 2

MODEL_CHOICE = Model.VGG

if MODEL_CHOICE == Model.VGG:
    from transfer_model import VGG_transfer_model as network_model, MODEL_NAME
elif MODEL_CHOICE == Model.HOMEBREW:
    from homebrew_model import homebrew_model as network_model, MODEL_NAME
else:
    raise RuntimeError("No model selected")

#either use latest or provide path to model below
USE_LATEST = True

dataset_dir = "MaskDataset"

if(USE_LATEST):
    base_model_exp_dir = f'experiments/{MODEL_NAME}'
    saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
    latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
    weights = os.path.join(latest_saved_weights_path, 'best/model')
else:
    weights = 'experiments/MaskDetection-Transfer/Nov18_05-28-22/best/model'


print(f"Loading weights from model: {weights}...")
model = network_model(256, 256, len(classes))
model.load_weights(weights)

# for each image in test folder, calculate prediction and add to results

results = {}
image_filenames = get_files_in_directory(f"{dataset_dir}/test")

# make a square image while keeping aspect ratio and filling with fill_color
def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im

for image_name in image_filenames:
    img = Image.open(f"{dataset_dir}/test/{image_name}").convert('RGB')
    # NN input is 256, 256
    img = img.resize((256, 256))
    img_array = np.expand_dims(np.array(img), 0) 
    # Normalize
    img_array = img_array / 255.

    # Get prediction
    softmax = model.predict(x=img_array)
    # Get predicted class (index with max value)
    prediction = tf.argmax(softmax, 1)
    # Get tensor's value
    prediction = tf.keras.backend.get_value(prediction)[0]

    results[image_name] = prediction

create_csv(results, MODEL_NAME)
