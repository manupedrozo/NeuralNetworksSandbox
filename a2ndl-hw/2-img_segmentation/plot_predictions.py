import os
import tensorflow as tf
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from files_in_dir import get_files_in_directory

def plot_predictions(model, valid_dataset, num_classes):
    # Assign a color to each class
    evenly_spaced_interval = np.linspace(0, 1, 2)
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    iterator = iter(valid_dataset)

    fig, ax = plt.subplots(1, 3, figsize=(8, 8))
    image, target, _ = next(iterator)
    target = np.reshape(target[0], (image.shape[1],image.shape[2]))

    image = image[0]

    out_sigmoid = model.predict(x=tf.expand_dims(image, 0))

    # Get predicted class as the index corresponding to the maximum value in the vector probability
    # predicted_class = tf.cast(out_sigmoid > score_th, tf.int32)
    # predicted_class = predicted_class[0, ..., 0]
    predicted_class = tf.argmax(out_sigmoid, -1)

    predicted_class = predicted_class[0, ...]

    # Assign colors (just for visualization)
    target_img = np.zeros([target.shape[0], target.shape[1], 3])
    prediction_img = np.zeros([target.shape[0], target.shape[1], 3])

    target_img[np.where(target == 0)] = [0, 0, 0]
    for i in range(1, num_classes):
        target_img[np.where(target == i)] = np.array(colors[i-1])[:3] * 255

    prediction_img[np.where(predicted_class == 0)] = [0, 0, 0]
    for i in range(1, num_classes):
        prediction_img[np.where(predicted_class == i)] = np.array(colors[i-1])[:3] * 255

    ax[0].imshow(np.uint8(image))
    ax[1].imshow(np.uint8(target_img))
    ax[2].imshow(np.uint8(prediction_img))

    plt.show()

def plot_only(image, num_classes):
    # Assign a color to each class
    evenly_spaced_interval = np.linspace(0, 1, 2)
    colors = [cm.rainbow(x) for x in evenly_spaced_interval]

    fig = plt.figure()
    ax = fig.add_subplot()

    prediction_img = np.zeros([image.shape[0], image.shape[1], 3])
    
    prediction_img[np.where(image == 0)] = [0, 0, 0]
    for i in range(1, num_classes):
        prediction_img[np.where(image == i)] = np.array(colors[i-1])[:3] * 255

    ax.imshow(np.uint8(prediction_img))
    plt.show()



if __name__ == "__main__":
    from vgg_base import create_model, CustomDataset, MODEL_NAME
    from dataset_types import Subdataset, Species
    from tensorflow.keras.applications.vgg16 import preprocess_input 

    dasaset_base = "Development_Dataset/Training"
    SUBDATASET = Subdataset.BIPBIP.value
    SPECIES = Species.HARICOT.value

    img_h, img_w = 256, 256
    bs = 32

    dataset_dir = os.path.join(dasaset_base, SUBDATASET, SPECIES)

    dataset_valid = CustomDataset(
        dataset_dir, 'validation', 
        preprocessing_function=preprocess_input
    )

    valid_dataset = tf.data.Dataset.from_generator(
        lambda: dataset_valid,
        output_types=(tf.float32, tf.float32, tf.float32),
        output_shapes=([img_h, img_w, 3], [img_h * img_w, 1], [img_h * img_w])
    ).batch(bs).repeat()

    num_classes = 3

    # get weights
    base_model_exp_dir = f"experiments/{MODEL_NAME}/{SUBDATASET}/{SPECIES}"
    saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
    latest_saved_weights_path = max(saved_weights, key=os.path.getctime)
    weights = os.path.join(latest_saved_weights_path, 'best/model')

    print(weights)

    model = create_model(img_h, img_w, num_classes)
    model.load_weights(weights)
    plot_predictions(model, valid_dataset, num_classes)
