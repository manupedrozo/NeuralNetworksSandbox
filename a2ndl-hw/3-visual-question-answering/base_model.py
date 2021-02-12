import os
import numpy as np
from PIL import Image
import random
from datetime import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import preprocess_input 

MODEL_NAME = 'QUESTION-ANSWERING'

def vgg_model(img_h, img_w, num_classes):
    vgg = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(img_h, img_w, 3))

    for layer in vgg.layers:
        layer.trainable = False

    model = tf.keras.Sequential()
    model.add(vgg)
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=512, activation='relu'))
    model.add(tf.keras.layers.Dense(units=num_classes, activation='softmax'))

    return model


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
            preprocessing_function=preprocess_input, subdataset=subdataset
        )
        dataset_valid = CustomDataset(
            dataset_dir, 'validation', 
            preprocessing_function=preprocess_input, subdataset=subdataset
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

        #model.summary()

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
