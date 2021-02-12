import os
import numpy as np
from PIL import Image
from mask_colors import BACKGROUND_0, BACKGROUND_1, CROP, WEED

def read_rgb_mask(img_path):
    '''
    img_path: path to the mask file
    Returns the numpy array containing target values
    '''

    mask_img = Image.open(img_path)
    mask_arr = np.array(mask_img)

    new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)

    # Use RGB dictionary in 'RGBtoTarget.txt' to convert RGB to target
    new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_0, axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_1, axis=-1))] = 0
    new_mask_arr[np.where(np.all(mask_arr == CROP, axis=-1))] = 1
    new_mask_arr[np.where(np.all(mask_arr == WEED, axis=-1))] = 2

    return new_mask_arr


if __name__ == "__main__":

    # Read the example RGB mask and transform it into integer labels.

    mask = read_rgb_mask("starting_kit/predictions/rgb_mask_example.png")

    np.save("starting_kit/predictions/arr_mask_example.npy", mask)
