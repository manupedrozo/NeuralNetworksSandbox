from enum import Enum
import numpy as np
import os
from tokens import get_tokenizer, UNKNOWN_TOKEN
from labels_dict import labels_dict

EMBEDDINGS_PATH = 'cache/saved_embeddings.npy'
ANSWER_DISTANCE_MATRIX_PATH = 'cache/saved_answer_distance_matrix.npy'

class Dimensions(Enum):
    D_50 = 50
    D_100 = 100
    D_200 = 200
    D_300 = 300

def build_embeddings(word_index, dim):
    # len + 2 to take into account padding and unknown token
    embedding_matrix = np.zeros((len(word_index) + 2, dim.value))
    word_found_mask = np.zeros((len(word_index) + 2,), dtype='bool')
    num_words = len(word_index)
    words_read = 0

    # padding is always 0
    word_found_mask[0] = True

    accumulator = None
    with open(f'glove.6B/glove.6B.{dim.value}d.txt', 'r', encoding="utf8") as f:
        for line in f:
            if words_read >= num_words:
                break
            values = line.split()
            word = values[0]
            if word in word_index:
                coefs = np.array(values[1:], dtype='float32')
                if accumulator is None:
                    accumulator = np.zeros(coefs.shape)
                accumulator += coefs
                embedding_matrix[word_index[word]] = coefs
                word_found_mask[word_index[word]] = True
                words_read += 1

    # Compute embedding for unknown values as the average of all the vectors in the vocabulary
    unknown_vector = accumulator / words_read
    embedding_matrix[word_found_mask == False] = unknown_vector

    return embedding_matrix

def get_embeddings():
    if not os.path.exists(EMBEDDINGS_PATH):
        tokenizer = get_tokenizer()
        embedding_matrix = build_embeddings(tokenizer.word_index, Dimensions.D_50)
        np.save(EMBEDDINGS_PATH, embedding_matrix)

    embedding_matrix = np.load(EMBEDDINGS_PATH)
    return embedding_matrix

def answer_distance_matrix(embedding_dim):
    num_labels = len(labels_dict)
    words_read = 0
    embedding_matrix = np.zeros((num_labels, embedding_dim.value))
    word_found_mask = np.zeros((num_labels,), dtype='bool')
    res = np.zeros((num_labels, num_labels))

    accumulator = None
    with open(f'glove.6B/glove.6B.{embedding_dim.value}d.txt', 'r', encoding="utf8") as f:
        for line in f:
            if words_read >= num_labels:
                break
            values = line.split()
            word = values[0]
            if word in labels_dict:
                coefs = np.array(values[1:], dtype='float32')
                if accumulator is None:
                    accumulator = np.zeros(coefs.shape)
                accumulator += coefs
                embedding_matrix[labels_dict[word]] = coefs
                word_found_mask[labels_dict[word]] = True
                words_read += 1
    
    # Compute embedding for unknown values as the average of all the vectors in the vocabulary
    unknown_vector = accumulator / words_read
    embedding_matrix[word_found_mask == False] = unknown_vector

    for i in range(num_labels):
        for j in range(num_labels):
            res[i, j] = np.linalg.norm(embedding_matrix[i] - embedding_matrix[j])

    return res

def get_answer_distance_matrix():
    if not os.path.exists(ANSWER_DISTANCE_MATRIX_PATH):
        ans_dist_matrix = answer_distance_matrix(Dimensions.D_50)
        np.save(ANSWER_DISTANCE_MATRIX_PATH, ans_dist_matrix)

    ans_dist_matrix = np.load(ANSWER_DISTANCE_MATRIX_PATH)
    return ans_dist_matrix

if __name__ == "__main__":
    # token_to_embedding = get_embeddings()
    # print("num embeddings", len(token_to_embedding))

    # print('unknown vector:', token_to_embedding[UNKNOWN_TOKEN])

    ans_dist_matrix = get_answer_distance_matrix()
    yes_idx = 57
    no_idx = 33
    yellow_idx = 56
    woman_idx = 55
    zero_idx = 0
    one_idx = 1
    two_idx = 2
    three_idx = 3
    print("yes-no distance:", ans_dist_matrix[yes_idx, no_idx])
    print("yes-yellow distance:", ans_dist_matrix[yes_idx, yellow_idx])
    print("yes-woman distance:", ans_dist_matrix[yes_idx, woman_idx])
    print("yes-0 distance:", ans_dist_matrix[yes_idx, zero_idx])
    print("yes-1 distance:", ans_dist_matrix[yes_idx, one_idx])
    print("1-0 distance:", ans_dist_matrix[one_idx, zero_idx])
    print("1-2 distance:", ans_dist_matrix[one_idx, two_idx])
    print("1-3 distance:", ans_dist_matrix[one_idx, three_idx])
    print("2-3 distance:", ans_dist_matrix[two_idx, three_idx])
