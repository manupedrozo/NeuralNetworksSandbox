import os
import numpy as np
from PIL import Image
from dataset_types import Subdataset, Species
import matplotlib.pyplot as plt
import math

def len_patches(image_shape, patch_size):
    """Return the amount of patches that can be extracted from an image"""
    patch_dim_x = image_shape[0] // patch_size[0]
    patch_dim_y = image_shape[1] // patch_size[1]
    return patch_dim_x * patch_dim_y

def get_patch(idx, image_array, patch_size):
    """Extract patch of a given size from an image array. idx must be in the range of [0, len_patches - 1]."""
    patch_dim_x = image_array.shape[0] // patch_size[0]
    patch_dim_y = image_array.shape[1] // patch_size[1]
    j = (idx % patch_dim_y) * patch_size[1]
    i = (idx // patch_dim_y) * patch_size[0]
    return image_array[i:patch_size[0]+i, j:patch_size[1]+j]

def generate_patches(image_array, patch_size):
    """Extract all patches of a given size from an image"""
    return np.stack([get_patch(i, image_array, patch_size) for i in range(len_patches(image_array.shape, patch_size))])

def restore_from_patches(patches, shape):
    """Join the patches into a single image of a given shape"""
    res = np.zeros(shape, dtype=np.int32)
    patch_size = patches.shape[1:]
    i = 0
    j = 0
    for p in patches:
        res[i:patch_size[0]+i, j:patch_size[1]+j] = p
        j += patch_size[1]
        if j + patch_size[1] > shape[1]:
            j = 0
            i += patch_size[0]
            if i + patch_size[0] > shape[0]:
                break

    return res

def calculate_padding(curr_size, desired_size):
    """Calculate how much to pad an image to achieve a desired size"""
    side_adjust = (desired_size - curr_size) / 2
    return (math.floor(side_adjust), math.ceil(side_adjust))

def get_shape_for_patching(img_shape, patch_size):
    """Calculate the needed size for an image so it can be divided evenly in patches of patch_size"""
    if img_shape[0] % patch_size[0] != 0:
        needed_x_size = math.ceil(img_shape[0] / patch_size[0]) * patch_size[0]
    else:
        needed_x_size = img_shape[0]

    if img_shape[1] % patch_size[1] != 0:
        needed_y_size = math.ceil(img_shape[1] / patch_size[1]) * patch_size[1]
    else:
        needed_y_size = img_shape[1]

    return (needed_x_size, needed_y_size)

def resize_for_patching(image_array, patch_size):
    """Resize a given image by padding so that it can be divided evenly in patches of patch_size"""
    img_shape = image_array.shape
    NO_PAD = (0,0)

    desired_shape = get_shape_for_patching(img_shape, patch_size)
    padding_x = calculate_padding(img_shape[0], desired_shape[0])
    padding_y = calculate_padding(img_shape[1], desired_shape[1])

    if padding_x == NO_PAD and padding_y == NO_PAD:
        return image_array
    else:
        paddings = [padding_x, padding_y]
        for i in range(len(img_shape) - len(paddings)):
            paddings.append(NO_PAD)
        return np.pad(image_array, paddings)

def remove_padding(image_array, original_shape):
    """Remove padding from an image so that it is restored to its original shape"""
    if image_array.shape == original_shape:
        return image_array

    padding_x = calculate_padding(original_shape[0], image_array.shape[0])
    padding_y = calculate_padding(original_shape[1], image_array.shape[1])

    return image_array[padding_x[0]:-padding_x[1], padding_y[0]:-padding_y[1]]

if __name__ == "__main__":
    patch_size = (256, 256)
    bipbip_shape = (2048, 1536)
    pead_shape = (3280, 2464)
    roseau_shape = (1228, 819)
    weedelec_shape = (5184, 3456)

    def test_patching(shape):
        dims_og = np.arange(shape[0] * shape[1]).reshape(shape)
        dims = resize_for_patching(dims_og, patch_size)
        patches = generate_patches(dims, patch_size)
        original = restore_from_patches(patches, dims.shape)
        np.testing.assert_array_equal(dims, original)
        original_og = remove_padding(original, shape)
        np.testing.assert_array_equal(dims_og, original_og)

    test_patching(bipbip_shape)
    test_patching(pead_shape)
    test_patching(roseau_shape)
    test_patching(weedelec_shape)

    print("Patching tests were successful")

    dataset_base = "Development_Dataset/Training"
    SUBDATASET = Subdataset.PEAD.value
    SPECIES = Species.HARICOT.value

    dataset_dir = os.path.join(dataset_base, SUBDATASET, SPECIES)
    image_name = "Pead_haricot_00006_i750"
    img_path = os.path.join(dataset_dir, "Images", f"{image_name}.jpg")

    img = Image.open(img_path).convert('RGB')
    img_array = np.array(img)
    patch_size = (256, 256)
    img_array_resized = resize_for_patching(img_array, patch_size)
    patches = generate_patches(img_array_resized, patch_size)

    reconstructed = restore_from_patches(patches, img_array_resized.shape)
    np.testing.assert_array_equal(img_array_resized, reconstructed)

    fig, ax = plt.subplots(1, 4, figsize=(8, 8))
    ax[0].imshow(img_array_resized)
    ax[1].imshow(patches[0])
    ax[2].imshow(patches[1])
    ax[3].imshow(reconstructed)

    plt.show()

    