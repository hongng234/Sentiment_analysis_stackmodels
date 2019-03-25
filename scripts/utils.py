import pandas as pd
import copy
import os
import numpy as np
import re
import keras.backend as K

from tqdm import tqdm
from collections import defaultdict
from os.path import abspath
from spacy.lang.vi import Vietnamese
from constant import DEFAULT_MAX_LENGTH
from gensim.models.keyedvectors import KeyedVectors
from sklearn.metrics import f1_score
import string

from augment import similar_augment, create_sim_dict, similar_augment_from_sim_dict
from sklearn.model_selection import train_test_split


def split_array(arr, condition):
    if len(arr) == 0:
        return []
    
    result = []
    accumulated = [arr[0]]

    for ele in arr[1:]:
        if condition(ele):
            result.append(copy.deepcopy(accumulated))
            accumulated = [copy.deepcopy(ele)]
        else:
            accumulated.append(copy.deepcopy(ele))
    
    result.append(copy.deepcopy(accumulated))
    return result

def read_file(file_path, is_train=True):
    file_path = abspath(file_path)
    data_lines = list(filter(lambda x: x != '', open(file_path, encoding='utf8').read().split('\n')))
    pattern = ('train' if is_train else 'test') + '_[0-9]{5}'
    datas = split_array(data_lines, lambda x: bool(re.match(pattern, x)))

    if is_train:
        result_array = list(map(lambda x: [x[0], ' '.join(x[1:-1]), int(x[-1])], datas))
    else:
        result_array = list(map(lambda x: [x[0], ' '.join(x[1:])], datas))

    columns = ['name', 'text', 'label'] if is_train else ['name', 'text']
    return pd.DataFrame(result_array, columns=columns)

def tokenize(texts):
    nlp = Vietnamese()
    docs = []

    for text in texts:
        tokens = np.array([postprocess_token(token.text) for token in nlp(text.lower())[1:-1]])
        docs.append(tokens)

    return docs

def postprocess_token(token):
    if token in string.punctuation:
        return '<punct>'
    elif token.isdigit():
        return '<number>'
    else:
        return token


def make_embeddings(texts, embedding_path, max_features):
    embedding_path = abspath(embedding_path)

    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    
    if embedding_path.endswith('.vec'):
        embedding_index = dict(get_coefs(*o.strip().split(' ')) for o in open(embedding_path, encoding='utf8'))
        mean_embedding = np.mean(np.array(list(embedding_index.values())))
    elif embedding_path.endswith('bin'):
        embedding_index = KeyedVectors.load_word2vec_format(embedding_path, binary=True)
        mean_embedding = np.mean(embedding_index.vectors, axis=0)

    embed_size = mean_embedding.shape[0]
    word_index = sorted(list({word.lower() for sentence in texts for word in sentence}))
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.zeros((nb_words + 1, embed_size))
    i = 1
    word_map = defaultdict(lambda: nb_words)

    for word in word_index:
        if i >= max_features:
            continue
        if word in embedding_index:
            embedding_matrix[i] = embedding_index[word]
        else:
            embedding_matrix[i] = mean_embedding
        word_map[word] = i
        i += 1

    embedding_matrix[-1] = mean_embedding
    return embed_size, word_map, embedding_matrix    


def text_to_sequences(texts, word_map, max_len=DEFAULT_MAX_LENGTH):
    texts_id = []
    for sentence in texts:
        sentence = [word_map[word.lower()] for word in sentence][:max_len]
        padded_sentence = np.pad(sentence, (0,max(0, max_len - len(sentence))), 'constant', constant_values=0)
        texts_id.append(padded_sentence)

    return np.array(texts_id)

def find_threshold(pred_proba, y_true, metric=f1_score):
    cur_acc = 0
    cur_thres = 0
    for ind in range(len(pred_proba) - 1):
        threshold = (pred_proba[ind][0] + pred_proba[ind + 1][0]) / 2
        pred = (pred_proba > threshold).astype(np.int8)
        acc = metric(pred, y_true)
        if acc > cur_acc:
            cur_thres = threshold
            cur_acc = acc

    return cur_thres


def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """
        Recall metric
        param y_true: groundtruth
        param y_pred: predictions
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """
        precision metric
        param y_true: groundtruth
        param y_pred: predictions
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall) / (precision+recall+K.epsilon()))


if __name__ == "__main__":
    
    train_file_path = '../data/train.crash'
    test_file_path = '../data/test.crash'
    train_data = read_file(train_file_path, is_train=True)
    test_data = read_file(test_file_path, is_train=False)


    train_data_tokenized = tokenize(train_data['text'])
    test_data_tokenized = tokenize(test_data['text'])
    labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)
    print(train_data_tokenized[10])
    print(labels[10])
    train_data_tokenized, val_data_tokenized, labels_train, labels_val = train_test_split(train_data_tokenized, labels, test_size=0.05)

    embedding_path = '../embeddings/smallFasttext.vi.vec'
    #train_tokenized_texts, train_labels = similar_augment(train_data_tokenized, labels, n_increase=0, model_path=embedding_path, n_word_replace=10, use_annoy=True, annoy_path=None)
    #print(train_tokenized_texts, train_labels)

    embed_size, word_map, embedding_mat = make_embeddings(list(train_data_tokenized) + list(val_data_tokenized) + list(test_data_tokenized), embedding_path, max_features=12000)
    