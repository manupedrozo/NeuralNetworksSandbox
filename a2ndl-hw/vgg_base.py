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

MODEL_NAME = 'Segmentation-transfer'

class CustomDataset(tf.keras.utils.Sequence):
    """
        CustomDataset inheriting from tf.keras.utils.Sequence.

        3 main methods:
        - __init__: save dataset params like directory, filenames..
        - __len__: return the total number of samples in the dataset
        - __getitem__: return a sample from the dataset

        Note: 
        - the custom dataset returns a single sample from the dataset. Then, we use 
            a tf.data.Dataset object to group samples into batches.
        - in this case we have a different structure of the dataset in memory. 
            We have all the images in the same folder and the training and validation splits
            are defined in text files.
    """

    def __init__(self, dataset_dir, which_subset, img_generator=None, mask_generator=None, 
        preprocessing_function=None, out_shape=[256, 256]):

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

    def __len__(self):
        return len(self.subset_filenames)

    def __getitem__(self, index):
        # Read Image
        curr_filename = self.subset_filenames[index]
        img = Image.open(os.path.join(self.dataset_dir, 'Images', curr_filename + '.jpg'))
        mask = Image.open(os.path.join(self.dataset_dir, 'Masks', curr_filename + '.png'))
        
        # Resize image and mask
        img = img.resize(self.out_shape)
        mask = mask.resize(self.out_shape, resample=Image.NEAREST)
        
        img_arr = np.array(img)
        mask_arr = np.array(mask)

        # Convert RGB mask for each class to numbers from 0 to 2
        new_mask_arr = np.zeros(mask_arr.shape[:2], dtype=mask_arr.dtype)
        new_mask_arr = np.expand_dims(new_mask_arr, -1)

        new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_0, axis=-1))] = 0
        new_mask_arr[np.where(np.all(mask_arr == BACKGROUND_1, axis=-1))] = 0
        new_mask_arr[np.where(np.all(mask_arr == CROP, axis=-1))] = 1
        new_mask_arr[np.where(np.all(mask_arr == WEED, axis=-1))] = 2

        if self.which_subset == 'training' and self.img_generator is not None and self.mask_generator is not None:
            # Perform data augmentation
            # We can get a random transformation from the ImageDataGenerator using get_random_transform
            # and we can apply it to the image using apply_transform
            transform_seed = random.randrange(0, 1 << 24)
            img_t = self.img_generator.get_random_transform(img_arr.shape, seed=transform_seed)
            mask_t = self.mask_generator.get_random_transform(mask_arr.shape, seed=transform_seed)
            img_arr = self.img_generator.apply_transform(img_arr, img_t)
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
            img_arr = self.preprocessing_function(img_arr)

        return img_arr, np.float32(out_mask)

# ---- Model ----
def create_model(img_h, img_w, num_classes):
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

    dasaset_base = "Development_Dataset/Training"

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
    bs = 32
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

        dataset = CustomDataset(
            dataset_dir, 'training', 
            img_generator=img_data_gen, mask_generator=mask_data_gen,
            preprocessing_function=preprocess_input
        )
        dataset_valid = CustomDataset(
            dataset_dir, 'validation', 
            preprocessing_function=preprocess_input
        )

        train_dataset = tf.data.Dataset.from_generator(
            lambda: dataset,
            output_types=(tf.float32, tf.float32),
            output_shapes=([img_h, img_w, 3], [img_h, img_w, 1])
        ).batch(bs).repeat()

        valid_dataset = tf.data.Dataset.from_generator(
            lambda: dataset_valid,
            output_types=(tf.float32, tf.float32),
            output_shapes=([img_h, img_w, 3], [img_h, img_w, 1])
        ).batch(bs).repeat()

        num_classes = 3
        model = create_model(img_h, img_w, num_classes=num_classes)

        model.summary()

        # Loss
        # Sparse Categorical Crossentropy to use integers (mask) instead of one-hot encoded labels
        loss = tf.keras.losses.SparseCategoricalCrossentropy() 

        # learning rate
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        # Validation metrics
        metrics = ['accuracy', gen_meanIoU(num_classes)]

        # Compile Model
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

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
            steps_per_epoch=len(dataset),
            validation_data=valid_dataset,
            validation_steps=len(dataset_valid), 
            callbacks=callbacks_list
        )

        if PLOT:
            if best_checkpoint_path:
                model.load_weights(best_checkpoint_path)
            # ---- Prediction ----
            plot_predictions(model, valid_dataset, num_classes)

    if TRAIN_ALL:
        for subdataset in Subdataset:
            for species in Species:
                # Training and validation datasets
                train_model(subdataset.value, species.value)
    else:
        train_model(subdataset.value, species.value)
