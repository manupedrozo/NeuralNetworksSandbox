import os
import numpy as np
from PIL import Image
import random
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib import cm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input 

import callbacks
from dataset_types import Subdataset, Species
from mask_colors import BACKGROUND_1, BACKGROUND_0, WEED, CROP
from plot_predictions import plot_predictions
from metrics import gen_meanIoU
import json
import patching
import math

MODEL_NAME = 'Segmentation-transfer'

# For ROSEAU, images are png, quick fix
def get_img_extension(subdataset):
    return '.png' if subdataset == Subdataset.ROSEAU.value else '.jpg'

class PatchedDataset(tf.keras.utils.Sequence):
    def __init__(self, dataset_dir, which_subset, img_generator=None, mask_generator=None, 
        preprocessing_function=None, out_shape=[256, 256], subdataset=None, img_scale=1):

        # Weigh classes according to their representation in the dataset.
        with open(os.path.join(dataset_dir, 'weights.json')) as f:
            self.class_weights = json.load(f)

        if which_subset == 'training':
            subset_file = os.path.join(dataset_dir, 'Splits', 'train.txt')
        elif which_subset == 'validation':
            subset_file = os.path.join(dataset_dir, 'Splits', 'val.txt')
        
        with open(subset_file, 'r') as f:
            lines = f.readlines()
        
        subset_filenames = []
        for line in lines:
            subset_filenames.append(line.strip()) 

        self.which_subset = which_subset
        self.dataset_dir = dataset_dir
        self.subset_filenames = subset_filenames
        self.img_generator = img_generator
        self.mask_generator = mask_generator
        self.preprocessing_function = preprocessing_function
        self.out_shape = out_shape
        self.subdataset = subdataset
 
        img = Image.open(os.path.join(self.dataset_dir, 'Images', self.subset_filenames[0] + get_img_extension(self.subdataset)))
        self.img_shape = (math.floor(img.size[0] * img_scale), math.floor(img.size[1] * img_scale))

        self.patches_per_img = patching.len_patches(patching.get_shape_for_patching(self.img_shape, self.out_shape), self.out_shape)

        # Shuffle the indexes so that the dataset generates random patches from different input images.
        self.indexes = np.arange(len(self))
        random.shuffle(self.indexes)

    def __len__(self):
        return len(self.subset_filenames) * self.patches_per_img

    def __getitem__(self, index):
        actual_index = self.indexes[index]

        # Read Image
        curr_filename = self.subset_filenames[actual_index // self.patches_per_img]
        patch_idx = actual_index % self.patches_per_img

        img = Image.open(os.path.join(self.dataset_dir, 'Images', curr_filename + get_img_extension(self.subdataset)))
        mask = Image.open(os.path.join(self.dataset_dir, 'Masks', curr_filename + '.png'))

        # Resize image and mask to reduce the number of patches and speed up training
        img = img.resize(self.img_shape)
        mask = mask.resize(self.img_shape, resample=Image.NEAREST)

        img_arr = patching.resize_for_patching(np.array(img), self.out_shape)
        mask_arr = patching.resize_for_patching(np.array(mask), self.out_shape)

        # get patch
        img_patch = patching.get_patch(patch_idx, img_arr, self.out_shape)
        mask_patch = patching.get_patch(patch_idx, mask_arr, self.out_shape)

        # Convert RGB mask for each class to numbers from 0 to 2
        new_mask_arr = np.zeros(mask_patch.shape[:2], dtype=mask_patch.dtype)
        new_mask_arr = np.expand_dims(new_mask_arr, -1)

        new_mask_arr[np.where(np.all(mask_patch == BACKGROUND_0, axis=-1))] = 0
        new_mask_arr[np.where(np.all(mask_patch == BACKGROUND_1, axis=-1))] = 0
        new_mask_arr[np.where(np.all(mask_patch == CROP, axis=-1))] = 1
        new_mask_arr[np.where(np.all(mask_patch == WEED, axis=-1))] = 2

        if self.which_subset == 'training' and self.img_generator is not None and self.mask_generator is not None:
            # Perform data augmentation
            # We can get a random transformation from the ImageDataGenerator using get_random_transform
            # and we can apply it to the image using apply_transform
            transform_seed = random.randrange(0, 1 << 24)
            img_t = self.img_generator.get_random_transform(img_patch.shape, seed=transform_seed)
            mask_t = self.mask_generator.get_random_transform(mask_patch.shape, seed=transform_seed)
            img_patch = self.img_generator.apply_transform(img_patch, img_t)
            # ImageDataGenerator use bilinear interpolation for augmenting the images.
            # Thus, when applied to the masks it will output 'interpolated classes', which
            # is an unwanted behaviour. As a trick, we can transform each class mask 
            # separately and then we can cast to integer values (as in the binary segmentation notebook).
            # Finally, we merge the augmented binary masks to obtain the final segmentation mask.
            out_mask = np.zeros_like(new_mask_arr)
            for c in np.unique(new_mask_arr):
                if c > 0:
                    curr_class_arr = np.float32(new_mask_arr == c)
                    curr_class_arr = self.mask_generator.apply_transform(curr_class_arr, mask_t)
                    # from [0, 1] to {0, 1}
                    curr_class_arr = np.uint8(curr_class_arr)
                    # recover original class
                    curr_class_arr = curr_class_arr * c 
                    out_mask += curr_class_arr
        else:
            out_mask = new_mask_arr
        
        if self.preprocessing_function is not None:
            img_patch = self.preprocessing_function(img_patch)

        out_mask = out_mask.reshape((self.out_shape[0] * self.out_shape[1], 1))

        # Weigh each pixel according to the target class weight
        weights = np.ndarray.flatten(out_mask)
        for c in range(len(self.class_weights)):
            weights[weights == c] = self.class_weights[c]

        return img_patch, np.float32(out_mask), weights

# ---- Model ----
def create_model(img_h, img_w, num_classes, train_mode=False):
    # Encoder
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

    for layer in vgg.layers:
        layer.trainable = False

    x = vgg.output

    skips = [
        vgg.get_layer('block1_pool'),
        vgg.get_layer('block2_pool'),
        vgg.get_layer('block3_pool'),
        vgg.get_layer('block4_pool'),
    ]
    
    start_f = 512
        
    depth = 5

    # Decoder
    for i in range(depth):
        x = tf.keras.layers.Conv2DTranspose(
            filters=start_f,
            kernel_size=(3, 3),
            strides=(2, 2),
            padding='same'
        )(x)
        skip_index = len(skips) - i - 1
        if skip_index >= 0:
            x = tf.keras.layers.Add()([x, skips[skip_index].output])
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)

        start_f = start_f // 2

    # Prediction Layer
    x = tf.keras.layers.Conv2D(
        filters=num_classes,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='softmax'
    )(x)

    if train_mode:
        x = tf.keras.layers.Reshape((img_h * img_w, num_classes))(x)
    
    return tf.keras.Model(inputs = vgg.input, outputs = x)

def set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == "__main__":
    # Set global seed for all internal generators, this should make all randomization reproducible
    import signal
    SEED = signal.SIGSEGV.value # Set SEED to SEG_FAULT code (11)
    set_seeds(SEED)

    dasaset_base = "Merged_Dataset/Training"

    TENSORBOARD = False
    CHECKPOINTS = False
    SAVE_BEST = True
    EARLY_STOP = True
    TRAIN_ALL = True
    AUGMENT_DATA = True

    if TRAIN_ALL:
        PLOT = False
    else: 
        subdataset = Subdataset.BIPBIP
        species = Species.HARICOT
        PLOT = True

    img_h = 256
    img_w = 256

    # Hyper parameters
    bs = 64
    lr = 1e-3
    epochs = 100

    # ---- ImageDataGenerator ----
    # Create training ImageDataGenerator object
    # We need two different generators for images and corresponding masks
    data_gen_parameters = {
        'rotation_range': 10,
        'width_shift_range': 20,
        'height_shift_range': 20,
        'zoom_range': 0.3,
        'horizontal_flip': True,
        'vertical_flip': True,
        'fill_mode': 'reflect'
    }

    if AUGMENT_DATA:
        img_data_gen = ImageDataGenerator(**data_gen_parameters)
        mask_data_gen = ImageDataGenerator(**data_gen_parameters)
    else:
        img_data_gen, mask_data_gen = None, None

    now = datetime.now().strftime('%b%d_%H-%M-%S')

    def train_model(subdataset, species):
        dataset_dir = os.path.join(dasaset_base, subdataset, species)

        dataset = PatchedDataset(
            dataset_dir, 'training', 
            img_generator=img_data_gen, mask_generator=mask_data_gen,
            preprocessing_function=preprocess_input, subdataset=subdataset
        )
        dataset_valid = PatchedDataset(
            dataset_dir, 'validation', 
            preprocessing_function=preprocess_input, subdataset=subdataset
        )

        train_dataset = tf.data.Dataset.from_generator(
            lambda: dataset,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=([img_h, img_w, 3], [img_h * img_w, 1], [img_h * img_w])
        ).batch(bs).repeat()

        valid_dataset = tf.data.Dataset.from_generator(
            lambda: dataset_valid,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=([img_h, img_w, 3], [img_h * img_w, 1], [img_h * img_w])
        ).batch(bs).repeat()

        num_classes = 3
        model = create_model(img_h, img_w, num_classes=num_classes, train_mode=True)

        if not TRAIN_ALL:
            model.summary()

        # Loss
        # Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
        loss = tf.keras.losses.SparseCategoricalCrossentropy() 

        # learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Validation metrics
        metrics = ['accuracy', gen_meanIoU(num_classes)]

        # Compile Model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics, sample_weight_mode="temporal")

        # ---- Callbacks ----
        exps_dir = "experiments"
        if not os.path.exists(exps_dir):
            os.makedirs(exps_dir)

        model_dir = os.path.join(exps_dir, MODEL_NAME)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        exp_dir = os.path.join(model_dir, subdataset, species, str(now))
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        callbacks_list = []

        # Model checkpoint
        if CHECKPOINTS:
            callbacks_list.append(callbacks.checkpoints(exp_dir))

        # Early stopping
        if EARLY_STOP:
            callbacks_list.append(callbacks.early_stopping(patience=10))

        # Save best model
        # ----------------
        best_checkpoint_path = None
        if SAVE_BEST:
            best_checkpoint_path, save_best_callback = callbacks.save_best(exp_dir)
            callbacks_list.append(save_best_callback)

        model.fit(
            x=train_dataset,
            epochs=epochs,
            steps_per_epoch=len(dataset) // bs,
            validation_data=valid_dataset,
            validation_steps=len(dataset_valid) // bs,
            callbacks=callbacks_list,
        )

        if PLOT:
            # rebuild model with no final reshaping
            prediction_model = create_model(img_h, img_w, num_classes=num_classes, train_mode=False)
            if best_checkpoint_path:
                prediction_model.load_weights(best_checkpoint_path).expect_partial()
            else:
                prediction_model.set_weights(model.get_weights())
            # ---- Prediction ----
            plot_predictions(prediction_model, valid_dataset, num_classes)
    
    c = 1
    if TRAIN_ALL:
        for subdataset in Subdataset:
            for species in Species:
                # Training and validation datasets
                print(f"Training {c}/8: {subdataset.value}/{species.value}...")
                c += 1
                train_model(subdataset.value, species.value)
    else:
        train_model(subdataset.value, species.value)
