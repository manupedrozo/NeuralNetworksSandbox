import os
import sys
import numpy as np
import copy
from PIL import Image

out_shape = [128, 128]

dataset_dir = "Development_Dataset/Training/Pead/Haricot"
curr_filename = "Pead_haricot_00006_i750"
img = Image.open(os.path.join(dataset_dir, 'Images', curr_filename + '.jpg'))
mask = Image.open(os.path.join(dataset_dir, 'Masks', curr_filename + '.png'))

# Resize image and mask
img = img.resize(out_shape)
mask = mask.resize(out_shape, resample=Image.NEAREST)

img_arr = np.array(img)
mask_arr = np.array(mask)

np.set_printoptions(threshold=sys.maxsize)

mask_arr2 = copy.deepcopy(mask_arr)

r1, g1, b1 = 254, 124, 18 # Original value
r2, g2, b2 = 0, 0, 0 # Value that we want to replace it with

red, green, blue = mask_arr[:,:,0], mask_arr[:,:,1], mask_arr[:,:,2]
color_mask = (red == r1) & (green == g1) & (blue == b1)
mask_arr[:,:,:3][color_mask] = [r2, g2, b2]

im = Image.fromarray(mask_arr)
im2 = Image.fromarray(mask_arr2)
im.show()
im2.show()


print(np.array_equal(mask_arr, mask_arr2))


