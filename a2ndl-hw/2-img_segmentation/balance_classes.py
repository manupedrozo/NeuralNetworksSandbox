import os
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight
from mask_colors import BACKGROUND_1, BACKGROUND_0, WEED, CROP
from dataset_types import Subdataset, Species
from files_in_dir import get_files_in_directory
from progressbar import progressbar
import json

"""
Read all masks for each dataset and calculate the average percentage of pixels belonging to each class in order to determine 
the weight to assign to each class. The less represented the class the higher the weight.
"""

def normalize_weights(weights):
    mininum = min(weights)
    for i in range(len(weights)):
        weights[i] /= mininum

if __name__ == "__main__":
    base_dir = 'Merged_Dataset/Training'
    for subdataset in Subdataset:
        for species in Species:
            print(f"{subdataset.value} | {species.value}:")
            dataset_dir = os.path.join(base_dir, subdataset.value, species.value)
            weights_file = os.path.join(dataset_dir, 'weights.json')

            if os.path.isfile(weights_file):
                print('Weigths already calculated:')
                with open(weights_file) as f:
                    weights = json.load(f)
                    print(weights)
                continue

            mask_dir = os.path.join(dataset_dir, 'Masks')
            weights_accumulator = None
            for filename in progressbar(get_files_in_directory(mask_dir)):
                mask = Image.open(os.path.join(mask_dir, filename))
                mask_arr = np.array(mask)

                new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)
                new_mask_arr = np.expand_dims(new_mask_arr, -1)
                new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_0, axis=-1))] = 0
                new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_1, axis=-1))] = 0
                new_mask_arr[np.where(np.all(mask_arr == CROP, axis=-1))] = 1
                new_mask_arr[np.where(np.all(mask_arr == WEED, axis=-1))] = 2

                # some masks may not have one of the classes (probably only weed may not appear)
                classes = []
                for i in [0, 1, 2]:
                    if i in new_mask_arr:
                        classes.append(i)

                weights = compute_class_weight('balanced', classes=classes, y=new_mask_arr.flatten())
                if weights_accumulator is not None:
                    # weights length will vary depending on the number of classes, weights_accmulator is always length 3
                    for c, i in enumerate(classes):
                        weights_accumulator[i] = (weights_accumulator[i] + weights[c]) / 2
                else:
                    weights_accumulator = np.zeros((3,))
                    for c, i in enumerate(classes):
                        weights_accumulator[i] = weights[c]

            weights = weights_accumulator.tolist()
            normalize_weights(weights)
            print(f"Weights:", weights)
            with open(weights_file, 'w') as f:
                json.dump(weights, f)
            print("Weights saved to:", weights_file)