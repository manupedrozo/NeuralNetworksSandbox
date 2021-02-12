import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # set TF logging to ERROR, needs to be done before importing TF
from datetime import datetime
from labels_dict import labels_dict
from files_in_dir import get_files_in_directory
from vqa_model import create_model, MODEL_NAME
from tokens import get_tokenizer, preprocess_question
from glove import get_embeddings
import numpy as np
from PIL import Image
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input 
import tensorflow.keras.backend as K
from progressbar import progressbar

def create_csv(results, results_dir='./'):
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')


def build_data(dataset_dir):
    # Create Tokenizer to convert words to integers
    questions = []
    images = []
    annotation_ids = []
    with open(os.path.join(dataset_dir, 'test_questions.json')) as f:
        annotations = json.load(f)
        for a_id, a in annotations.items():
            questions.append(preprocess_question(a['question']))
            images.append(a['image_id'] + ".png")
            annotation_ids.append(a_id)
    
    tokenizer = get_tokenizer()
    tokenized = tokenizer.texts_to_sequences(questions)
    text_inputs = pad_sequences(tokenized, maxlen=max_seq_length)
    
    return annotation_ids, text_inputs, images

dataset_dir = 'VQA_Dataset'

num_classes = len(labels_dict)
img_dim = 256
max_seq_length = 21

base_model_exp_dir = f"experiments/{MODEL_NAME}"
saved_weights = [os.path.join(base_model_exp_dir, f) for f in get_files_in_directory(base_model_exp_dir, include_folders=True)]
latest_saved_weights_root = max(saved_weights, key=os.path.getctime)
folds = [os.path.join(latest_saved_weights_root, f) for f in get_files_in_directory(latest_saved_weights_root, include_folders=True)]
annotation_ids, text_inputs, images = build_data(dataset_dir)
print(f"{annotation_ids[0]} {text_inputs[0]} {images[0]}")

results = {}
bs = 64

predictions = np.zeros((len(annotation_ids), num_classes))

for f_idx, f in enumerate(folds):
    print(f"predicting for fold {f_idx}")
    weights = os.path.join(f, 'best/model')

    model = create_model(img_dim, img_dim, 
        num_classes=num_classes, 
        max_seq_length=max_seq_length, 
        embedding_matrix=get_embeddings(),
    )

    model.load_weights(weights).expect_partial()

    for i in progressbar(range(0, len(annotation_ids), bs)):
        question_batch = np.array(text_inputs[i:i+bs])
        image_batch = images[i:i+bs]
        
        img_arrays = []
        for image in image_batch:
            img = Image.open(os.path.join(dataset_dir, 'Images', image)).convert('RGB')
            img = img.resize([img_dim, img_dim])
            img_arr = np.array(img)
            img_arr = preprocess_input(img_arr)
            img_arrays.append(img_arr)
        
        img_arrays = np.stack(img_arrays)
        softmax = model.predict([img_arrays, question_batch])
        predictions[i:i+bs] += softmax

    K.clear_session()
        
num_folds = len(folds)
predictions /= num_folds

for i in range(len(annotation_ids)):
    a_id = annotation_ids[i]
    prediction = tf.argmax(predictions[i])
    prediction = tf.keras.backend.get_value(prediction)
    results[a_id] = prediction

create_csv(results)
    








