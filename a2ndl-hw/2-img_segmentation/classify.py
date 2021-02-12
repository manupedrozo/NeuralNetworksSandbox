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
from tensorflow.keras.applications.vgg16 import preprocess_input 
from files_in_dir import get_files_in_directory
import patching

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

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

PLOT = False

patch_size = 256
num_classes = 3
# results dict
submission_dict = {}

# loop through subdatasets and species
for subdataset in Subdataset:
    SUBDATASET = subdataset.value
    for species in Species:
        SPECIES = species.value
        print(f"Classifying {SUBDATASET}/{SPECIES}")

        # dataset dir
        dataset_dir = f"Merged_Dataset/Test_Final/{SUBDATASET}/{SPECIES}/Images"
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

        model = create_model(patch_size, patch_size, num_classes)
        model.load_weights(weights).expect_partial()

        # calculate prediction for each image
        for img_name in image_filenames:
            img = Image.open(f"{dataset_dir}/{img_name}")
            img_width, img_height = img.size
            #img = img.resize([256, 256])

            img_array = np.array(img)
            
            padded_img = patching.resize_for_patching(img_array, (patch_size, patch_size))
            patches = patching.generate_patches(padded_img, (patch_size, patch_size))
            
            img_array = preprocess_input(img_array)

            predicted = []
            for p in patches:
                p = preprocess_input(p)

                # predict -> (256, 256) with class value
                patch_pred = model.predict(x=np.expand_dims(p, 0))
                patch_pred = tf.argmax(patch_pred, -1) 
                ## Get tensor's value
                patch_pred = tf.keras.backend.get_value(patch_pred).reshape((patch_size, patch_size))
                predicted.append(patch_pred)                
 

            predicted = np.stack(predicted)
            predicted = patching.restore_from_patches(predicted, padded_img.shape[:-1])
            prediction = patching.remove_padding(predicted, img_array.shape[:-1])

            if PLOT:
                plot_only(prediction, 3)

            #prediction = cv2.resize(np.uint8(prediction), dsize=(img_width, img_height), interpolation = cv2.INTER_NEAREST)

            #if PLOT:
            #    plot_only(prediction, 3)

            

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

