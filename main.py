import sys
sys.path.append('./scripts')
from utils import read_file, tokenize, make_embeddings, text_to_sequences, find_threshold
from augment import similar_augment, create_sim_dict, similar_augment_from_sim_dict
from constant import DEFAULT_MAX_FEATURES
from cnn import TextCNN, VDCNN, LSTMCNN
from rnn import RNNKeras, LSTMKeras, SARNNKeras, HARNN

import argparse
import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.vis_utils import plot_model


def train_model(model, embedding_path, annoy_path, max_features, should_find_threshold, should_mix, return_prob, trainable, use_additive_emb, augment_size, use_sim_dict, print_model, model_high):
    augment_size = int(augment_size)
    embedding_path = './embeddings/smallFasttext.vi.vec'

    train_file_path = './data/train.crash'
    test_file_path = './data/test.crash'
    train_data = read_file(train_file_path, is_train=True)
    test_data = read_file(test_file_path, is_train=False)
    train_tokenized_texts = tokenize(train_data['text'])
    test_tokenized_texts = tokenize(test_data['text'])
    labels = train_data['label'].values.astype(np.float16).reshape(-1,1)


    train_tokenized_texts, val_tokenized_texts, labels_train, labels_val = train_test_split(train_tokenized_texts, labels, test_size=0.05)
    labels_test = test_data['label'].values.astype(np.float16).reshape(-1,1)

    print(train_tokenized_texts[10], labels[10])

    if augment_size != 0:
        if augment_size < 0:
            augment_size = len(train_tokenized_texts) * (-augment_size)

        print(augment_size)

        train_tokenized_texts, labels_train = similar_augment(train_tokenized_texts, labels, n_increase=augment_size, model_path=embedding_path, n_word_replace=10, use_annoy=True, annoy_path=None)
        print(train_tokenized_texts)

    
    embed_size, word_map, embedding_mat = make_embeddings(list(train_tokenized_texts)+list(val_tokenized_texts)+list(test_tokenized_texts), embedding_path, max_features=max_features)

    texts_id_train = text_to_sequences(train_tokenized_texts, word_map)
    
    if augment_size != 0:
        if augment_size < 0:
            augment_size = len(train_tokenized_texts) * (-augment_size)
        sim_dict = create_sim_dict(word_map, model_path=embedding_path, annoy_path=None)
        print('Finnish creating sim dict')
        texts_id_train, labels_train = similar_augment_from_sim_dict(texts_id_train, labels_train, sim_dict, n_increase=augment_size)

    texts_id_val = text_to_sequences(val_tokenized_texts, word_map)
    print('\nNumber of train data: {}, Number of val data: {}'.format(labels_train.shape, labels_val.shape))


    model_path = './models'

    try:
        os.mkdir('./models')
    except:
        print('Folder already created!!')
    try:
        os.mkdir(model_path)
    except:
        print('Folder already created!!')
    
    checkpoint = ModelCheckpoint(filepath='{}/models.hdf5'.format(model_path), monitor='val_f1', verbose=1, mode='max', save_best_only=True)
    early = EarlyStopping(monitor='val_f1', mode='max', patience=5)
    callbacks_list = [checkpoint, early]
    
    batch_size = 16
    epochs = 2

    model = model(embeddingMatrix=embedding_mat, embed_size=embed_size, max_features=embedding_mat.shape[0], trainable=trainable, use_additive_emb=use_additive_emb)
    if print_model:
        plot_model(model, to_file='{}.jpg'.format(model_high), show_shapes=True, show_layer_names=True)
        return
    model.fit(texts_id_train, labels_train, validation_data=(texts_id_val, labels_val), callbacks=callbacks_list, epochs=epochs, batch_size=batch_size)


    model.load_weights('{}/models.hdf5'.format(model_path))
    prediction_prob = model.predict(texts_id_val)
    if should_find_threshold:
        OPTIMAL_THRESHOLD = find_threshold(prediction_prob, labels_val)
    else:
        OPTIMAL_THRESHOLD = 0.5
    print('OPTIMAL_THRESHOLD: {}'.format(OPTIMAL_THRESHOLD))
    
    prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
    print('F1 validation score: {}'.format(f1_score(prediction, labels_val)))
    with open('{}/f1'.format(model_path), 'w') as fp:
        fp.write(str(f1_score(prediction, labels_val)))

    test_id_texts = text_to_sequences(test_tokenized_texts, word_map)
    test_prediction = model.predict(test_id_texts)

    df_prediction = pd.read_csv('./data/sample_submission.csv')
    if return_prob:
        df_prediction['label'] = test_prediction
    else:
        df_prediction['label'] = (test_prediction > OPTIMAL_THRESHOLD).astype(np.int8)

    print('Number of test data: {}'.format(df_prediction.shape[0]))
    df_prediction.to_csv('{}/prediction.csv'.format(model_path), index=False)



model_dict = {
    'RNNKeras': RNNKeras,
    'LSTMKeras': LSTMKeras,
    'SARNNKeras': SARNNKeras,
    'TextCNN': TextCNN,
    'LSTMCNN': LSTMCNN,
    'VDCNN': VDCNN
}
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-m',
        '--model',
        help='Model use',
        default='RNNKeras'
    )
    parser.add_argument(
        '-e',
        '--embedding',
        help='Model use',
        default='./embeddings/smallFasttext.vi.vec'
    )
    parser.add_argument(
        '-annoy',
        '--annoy',
        help='Model use',
        default='./embeddings/annoy.pkl'
    )
    parser.add_argument(
        '--max',
        help='Model use',
        default=DEFAULT_MAX_FEATURES
    )
    parser.add_argument(
        '--aug',
        help='Model use',
        default=0
    )
    parser.add_argument(
        '--use_sim_dict',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--find_threshold',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--mix',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--prob',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--fix_embed',
        action='store_false',
        help='Model use'
    )
    parser.add_argument(
        '--add_embed',
        action='store_true',
        help='Model use'
    )
    parser.add_argument(
        '--print_model',
        action='store_true',
        help='Model use'
    )

    args = parser.parse_args()
    if not args.model in model_dict:
        raise RuntimeError('Model Not Found!!!')
    
    train_model(model_dict[args.model], args.embedding, args.annoy, int(args.max), args.find_threshold, args.mix, args.prob, args.fix_embed, args.add_embed, args.aug, args.use_sim_dict, args.print_model, args.model)