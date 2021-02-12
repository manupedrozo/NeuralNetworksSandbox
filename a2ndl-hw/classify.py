import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from vgg_base import create_model, MODEL_NAME
from dataset_types import Subdataset, Species
from tensorflow.keras import backend
from datetime import datetime
from plot_predictions import plot_only
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def get_files_in_directory(path, include_folders=False):
    """Get all filenames in a given directory, optionally include folders as well"""
    return [f for f in os.listdir(path) if include_folders or os.path.isfile(os.path.join(path, f))]


def rle_encode(img):
    '''
    img: numpy array, 1 - foreground, 0 - background
    Returns run length as string formatted
    '''
    pixels = np.asarray(img).flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)

def empty_prediction(img_filenames, subdataset, species):
    '''
    empty dict for each image
    '''
    for img_name in image_filenames:
        img_name = os.path.splitext(img_name)[0]
        submission_dict[img_name] = {}
        submission_dict[img_name]['shape'] = ""
        submission_dict[img_name]['team'] = subdataset
        submission_dict[img_name]['crop'] = species
        submission_dict[img_name]['segmentation'] = {}
        submission_dict[img_name]['segmentation']['crop'] = ""
        submission_dict[img_name]['segmentation']['weed'] = ""


# results dict
submission_dict = {}

# loop through subdatasets and species
for subdataset in Subdataset:
    SUBDATASET = subdataset.value
    for species in Species:
        SPECIES = species.value
        print(f"Classifying {SUBDATASET}/{SPECIES}")

        # dataset dir
        dataset_dir = f"Development_Dataset/Test_Dev/{SUBDATASET}/{SPECIES}/Images"
        image_filenames = get_files_in_directory(dataset_dir)

        # get weights
        base_model_exp_dir = f"experiments/{MODEL_NAME}/{SUBDATASET}/{SPECIES}"

        if not os.path.exists(base_model_exp_dir):
            print("\tNo experiments found, skipping...")
            empty_prediction(image_filenames, SUBDATASET, SPECIES)
            continue

        saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
        latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
        weights = os.path.join(latest_saved_weights_path, 'best/model')


        # load weights in model
        print(f"\tLoading weights from model: {weights}...")
        model = create_model(256, 256, 3)
        model.load_weights(weights)


        # calculate prediction for each image
        for img_name in image_filenames:
            img = Image.open(f"{dataset_dir}/{img_name}").convert('RGB')
            #img.show()
            img = img.resize((256, 256))
            img_array = np.expand_dims(np.array(img), 0) 

            # predict -> (256, 256) with class value
            prediction = model.predict(x=img_array)
            prediction = tf.argmax(prediction, -1) 
            ## Get tensor's value
            prediction = np.matrix(tf.keras.backend.get_value(prediction))


            #plot_only(prediction, 3)
            prediction = cv2.resize(np.uint8(prediction), dsize=(2048,1536), interpolation = cv2.INTER_NEAREST)
            #plot_only(prediction, 3)

            img_name = os.path.splitext(img_name)[0] # remove extension
            submission_dict[img_name] = {}
            submission_dict[img_name]['shape'] = prediction.shape
            submission_dict[img_name]['team'] = SUBDATASET
            submission_dict[img_name]['crop'] = SPECIES
            submission_dict[img_name]['segmentation'] = {}

            # RLE encoding
            # crop
            rle_encoded_crop = rle_encode(prediction == 1)
            # weed
            rle_encoded_weed = rle_encode(prediction == 2)

            submission_dict[img_name]['segmentation']['crop'] = rle_encoded_crop
            submission_dict[img_name]['segmentation']['weed'] = rle_encoded_weed


# Save results into the submission.json file
now = str(datetime.now().strftime('%b%d_%H-%M-%S'))
if len(submission_dict) > 0:
    predictions_dir = f"predictions/{now}"
    if not os.path.exists(predictions_dir):
        os.makedirs(predictions_dir)
    predictions_path = f"{predictions_dir}/submission.json"
    with open(predictions_path, 'w') as f:
        json.dump(submission_dict, f)
        print(f"Classification results at {predictions_path}")

