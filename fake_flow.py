#!/usr/bin/env python
# -*- coding: utf-8 -*-

import warnings
warnings.filterwarnings('ignore')

import os, random, gensim
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from os.path import exists, join
import numpy as np
from tqdm import tqdm
import pandas as pd
from nltk import tokenize
from argparse import ArgumentParser
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from read_data import prepare_input

# Keras
import keras
from keras import backend as K
from keras.layers import Dense, GRU, Embedding, Input, Dropout, Bidirectional, MaxPooling1D, Convolution1D, Flatten, Concatenate, concatenate, TimeDistributed
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from keras_self_attention import SeqSelfAttention
from hyperopt import fmin, tpe, hp, Trials
from hyperopt import STATUS_OK

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Reproducibility
import tensorflow as tf
np.random.seed(0)
random.seed(0)
tf.random.set_random_seed(0)




class fake_flow:

    def __init__(self, model_name, SHUFFLE=False):
        # Settings
        self.scoring = 'f1'
        self.verbose = 1
        self.SHUFFLE = SHUFFLE
        self.model_name = model_name
        self.search = False
        self.summary_table = {}
        self.labelencoder = LabelEncoder()
        self.labelencoder.fit(['a'])

        self.parameters = {'rnn_size': 8,
                           'activation_rnn': 'tanh',

                           'num_filters': 16,
                           'filter_sizes': (2, 3, 4),
                           'pool_size': 3,
                           'activation_cnn': 'relu',

                           'dense_1': 8,
                           'activation_attention': 'softmax',
                           'dense_2': 8,
                           'dense_3': 8,
                           'dropout': 0.3910,
                           'optimizer': 'adam',

                           'max_senten_len': 500,
                           'max_senten_num': int(model_name.split('_')[1]),
                           'vocab': 1000000,
                           'embedding_path': './GoogleNews-vectors-negative300.bin',
                           'embeddingSize': 300,
                           'max_epoch': 50,
                           'batch_size': 16,
                           }
        self.tokenizer = Tokenizer(num_words=self.parameters['vocab'], oov_token=True)

    def shuffle_along_axis(self, a):
        idx = [item for item in range(a['features'].shape[1])]
        idx = shuffle(idx, random_state=0)
        a['features'] = a['features'][:, idx, :]
        a['text'] = np.array([item.split('.') for item in a['text']])
        a['text'] = a['text'][:, idx]
        a['text'] = np.array([' . '.join(item) for item in a['text'].tolist()])
        return a

    def prepare_input(self, train, dev, test):
        if self.SHUFFLE == True:
            train = self.shuffle_along_axis(train)
            dev = self.shuffle_along_axis(dev)
            test = self.shuffle_along_axis(test)
            self.model_name += '_shuffled'

        self.train, self.dev, self.test = train, dev, test
        self.train['text'] = self.train['text'].tolist()
        self.dev['text'] = self.dev['text'].tolist()
        self.test['text'] = self.test['text'].tolist()

        self.tokenizer.fit_on_texts(self.train['text'] + self.dev['text'] + self.test['text'])
        self.parameters['vocab'] = len(self.tokenizer.word_counts) + 1

        self.train['text'], self.train['label'] = self.preprocessing(train['text'], train['label'])
        self.dev['text'], self.dev['label'] = self.preprocessing(dev['text'], dev['label'])
        self.test['text'], self.test['label'] = self.preprocessing(test['text'], test['label'])

        self.prep_embed()

    def preprocessing(self, text, label):
        """Preprocessing of the text to make it more resonant for training
        """
        paras = []
        max_sent_num = 0
        max_sent_len = 0
        for idx in tqdm(range(len(text)), desc='Tokenizing text'):
            sentences = tokenize.sent_tokenize(text[idx])
            if len(sentences) > 45:
                print()
            if len(sentences) > max_sent_num:
                max_sent_num = len(sentences)
            tmp = [len(sent.split()) for sent in sentences]
            sent_len = max(tmp)
            if sent_len > max_sent_len:
                max_sent_len = sent_len
            paras.append(sentences)

        data = np.zeros((len(text), self.parameters['max_senten_num'], self.parameters['max_senten_len']), dtype='int32')
        for i, sentences in tqdm(enumerate(paras), desc='Preparing input matrix'):
            for j, sent in enumerate(sentences):
                if j < self.parameters['max_senten_num']:
                    wordTokens = text_to_word_sequence(sent)
                    k = 0
                    for _, word in enumerate(wordTokens):
                        if k < self.parameters['max_senten_len'] and word in self.tokenizer.word_index and self.tokenizer.word_index[word] < self.parameters['vocab']:
                            data[i, j, k] = self.tokenizer.word_index[word]
                            k = k+1
        labels = self.preparing_labels(label)
        return data, labels

    def preparing_labels(self, y):
        y = np.array(y)
        y = y.astype(str)
        if y.dtype.type == np.array(['a']).dtype.type:
            if len(self.labelencoder.classes_) < 2:
                self.labelencoder.fit(y)
                self.Labels = self.labelencoder.classes_.tolist()
            y = self.labelencoder.transform(y)
        labels = to_categorical(y, len(self.Labels))
        return labels

    def prep_embed(self):
        path = './processed_files'
        filename = '{}.npy'.format(self.model_name.split('_')[0])
        file = exists(join(path, filename))
        if file:
            embedding_matrix = np.load(join(path, filename))
        else:
            embeddings_index = gensim.models.KeyedVectors.load_word2vec_format(self.parameters['embedding_path'], binary=True)
            embedding_matrix = np.zeros((self.parameters['vocab'], self.parameters['embeddingSize']))

            for word, i in tqdm(self.tokenizer.word_index.items()):
                if i >= self.parameters['vocab']:
                    continue
                if word in embeddings_index:  # .vocab
                    embedding_matrix[i] = embeddings_index[word]
            embeddings_index = {}
            np.save(join(path, filename), embedding_matrix)

        self.embedding_matrix = embedding_matrix

    def Network(self):
        Embed = Embedding(input_dim=self.parameters['vocab'],
                               output_dim=self.parameters['embeddingSize'],
                               input_length=self.parameters['max_senten_len'],
                               trainable=True,
                               weights=[self.embedding_matrix],
                               name='Embed_Layer')

        # Features:
        inp_features = Input(shape=(self.train['features'].shape[1], self.train['features'].shape[2]), name='features_input')
        flow_features = Bidirectional(GRU(self.parameters['rnn_size'], activation=self.parameters['activation_rnn'], return_sequences=True, name='rnn'))(inp_features)
        features = Model(inp_features, flow_features)
        if self.verbose == 1:
            features.summary()

        # WordEmbd
        word_input = Input(shape=(self.parameters['max_senten_len'],), dtype='float32')
        z = Embed(word_input)
        conv_blocks = []
        for sz in self.parameters['filter_sizes']:
            conv = Convolution1D(filters=self.parameters['num_filters'], kernel_size=sz, padding="valid", activation=self.parameters['activation_cnn'], strides=1)(z)
            conv = MaxPooling1D(pool_size=self.parameters['pool_size'])(conv)
            conv = Flatten()(conv)
            conv_blocks.append(conv)
        z = Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
        wordEncoder = Model(word_input, z)
        if self.verbose == 1:
            wordEncoder.summary()

        # Sentences Concated
        sent_input = Input(shape=(self.train['text'].shape[1], self.parameters['max_senten_len']), dtype='float32', name='input_2')
        y = TimeDistributed(wordEncoder, name='input_sent2')(sent_input)
        y = Dense(self.parameters['dense_1'], name='dense_1')(y)
        y = concatenate([inp_features, y], axis=2)
        y = Dense(self.parameters['dense_2'], name='dense_2')(y)
        y = SeqSelfAttention(attention_activation=self.parameters['activation_attention'], return_attention=False, name='Self-Attention')(y)

        y = keras.layers.dot([flow_features, y], axes=[1, 1])
        y = keras.layers.Lambda(lambda x: K.mean(x, axis=1))(y)
        y = Dense(self.parameters['dense_3'], name='dense_3')(y)
        y = Dropout(self.parameters['dropout'])(y)
        y = Dense(len(self.Labels), activation='softmax', name='final_softmax')(y)
        model = Model([inp_features, sent_input], y)
        if self.verbose == 1:
            model.summary()
        return model

    def evaluate_on_test(self):
        Y_test_pred = self.model.predict([self.test['features'], self.test['text']], batch_size=self.parameters['batch_size'], verbose=0)
        Y_test_pred = np.argmax(Y_test_pred, axis=1)
        Y_test = np.argmax(self.test['label'], axis=1)
        print(classification_report(Y_test, Y_test_pred))

    def run_model(self, type_='train'):
        file = './processed_files/saved_models/{}_{}.check'.format(self.__class__.__name__, self.model_name)
        # 1, Compile the model
        self.session = InteractiveSession(config=config)
        self.model = self.Network()
        self.model.compile(optimizer=self.parameters['optimizer'], loss='categorical_crossentropy', metrics=['accuracy'])

        # 2, Prep
        callback = [EarlyStopping(min_delta=0.0001, patience=4, verbose=2, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.03, patience=3, verbose=2, min_lr=0.00001)]
        if not self.search:
            callback.append(ModelCheckpoint(file, save_best_only=True, save_weights_only=False))

        # 3, Train
        if type_ == 'train':
            self.model.fit(x=[self.train['features'], self.train['text']], y=self.train['label'], batch_size=self.parameters['batch_size'], epochs=self.parameters['max_epoch'], verbose=self.verbose,
                           validation_data=([self.dev['features'], self.dev['text']], self.dev['label']), callbacks=callback)
        elif type_ == 'test':
            if os.path.exists(file):
                self.model.load_weights(file, by_name=True)
                print('--------Load Weights Successful!--------')
            else:
                print('Model doesn\'t exist: {}'.format(file[file.rfind('/'):]))
                exit(1)
        else:
            print('Mode not defined!')
            exit(1)

        # 4, Evaluate
        if not self.search:
            self.evaluate_on_test()
            if str(self.model_name).__contains__('truthShades'):
                print('In-domain test:')
                self.test = self.test_in
                self.evaluate_on_test()

    def run_hyperopt_search(self, n_evals):
        self.verbose = 0
        self.search = True
        # self.pbar = tqdm(total=n_evals)
        # self.pbar.set_description('Hyperopt evals')
        search_space = {'rnn_size': hp.choice('lstmSize', [8, 16, 32, 64, 128]),
                        'activation_rnn': hp.choice('activation_rnn', ['selu', 'relu', 'tanh', 'elu']),

                        'num_filters': hp.choice('num_filters', [4, 8, 16, 32, 64]),
                        'filter_sizes': hp.choice('filter_sizes', [(2, 3, 4), (3, 4, 5), (4, 5, 6), (3, 5), (2, 4), (4,), (5,), (3, 5, 7), (3, 6)]),
                        'activation_cnn': hp.choice('activation_cnn', ['selu', 'relu', 'tanh', 'elu']),
                        'pool_size': hp.choice('pool_size', [2, 3]),

                        'dense_1': hp.choice('dense_1', [8, 16, 32, 64, 128]),
                        'activation_attention': hp.choice('activation_attention', ['sigmoid', 'softmax']),
                        'dense_2': hp.choice('dense_2', [8, 16, 32, 64, 128]),
                        'dense_3': hp.choice('dense_3', [8, 16, 32, 64, 128]),
                        'dropout': hp.uniform('dropout', 0.1, 0.6),
                        'optimizer': hp.choice('optimizer', ['adam', 'adadelta', 'rmsprop', 'sgd']),
                        }
        trials = Trials()
        best = fmin(self.objective_function, space=search_space, algo=tpe.suggest, max_evals=n_evals, trials=trials)
        # self.pbar.close()
        bp = trials.best_trial['result']['Params']
        print('\n\n', best)
        print(bp)

    def objective_function(self, params):
        mean_score = self.Kstratified(params)
        params.update({'score': mean_score})
        params = {key: str(val) for key, val in params.items()}
        print(params)

        if len(self.summary_table) < 1:
            self.summary_table.update(params)
        else:
            for key, value in self.summary_table.items():
                if key in params:
                    values = self.summary_table[key]
                    if not type(values) is list:
                        values = [values]
                    values.append(params[key])
                    self.summary_table[key] = values

        try:
            df_summary_table = pd.DataFrame(self.summary_table, index=[0])
        except:
            df_summary_table = pd.DataFrame(self.summary_table)
        df_summary_table.sort_values('score', inplace=True, ascending=False)
        df_summary_table.to_csv('./processed_files/saved_models/results_{}_{}.csv'.format(self.__class__.__name__, self.model_name), header=True, index=False)

        output = {'loss': 1 - mean_score,
                  'Params': params,
                  'status': STATUS_OK,
                  }
        # self.pbar.update(1)
        return output

    def Kstratified(self, params):

        self.parameters['rnn_size'] = params['rnn_size']
        self.parameters['activation_rnn'] = params['activation_rnn']

        self.parameters['num_filters'] = params['num_filters']
        self.parameters['filter_sizes'] = params['filter_sizes']
        self.parameters['pool_size'] = params['pool_size']
        self.parameters['activation_cnn'] = params['activation_cnn']

        self.parameters['dense_1'] = params['dense_1']
        self.parameters['activation_attention'] = params['activation_attention']
        self.parameters['dense_2'] = params['dense_2']
        self.parameters['dense_3'] = params['dense_3']
        self.parameters['dropout'] = params['dropout']
        self.parameters['optimizer'] = params['optimizer']

        print('Current: {}'.format(params))
        self.run_model()
        Y_dev_pred = self.model.predict([self.dev['features'], self.dev['text']], batch_size=self.parameters['batch_size'], verbose=0)
        Y_dev_pred = np.argmax(Y_dev_pred, axis=1)
        self.Y_dev = np.argmax(self.dev['label'], axis=1)
        # self.session.close()

        if self.scoring.lower() == 'f1':
            return f1_score(self.Y_dev, Y_dev_pred, average='macro')
        elif self.scoring.lower() == 'acc':
            return accuracy_score(self.Y_dev, Y_dev_pred)

if __name__ == '__main__':
    dataset = 'MultiSourceFake'
    segments_number = 10
    search = 0
    type_ = "test"

    parser = ArgumentParser()
    parser.add_argument("-d", "--dataset", help="Dataset name", default=dataset)
    parser.add_argument("-sn", "--segments_number", help="Number of segments - the default value is 20", default=segments_number, type=int)
    parser.add_argument("-s", "--search", help="search for best parameters", default=search, type=int)
    parser.add_argument("-m", "--mode", help="train or test", default=type_)
    args = parser.parse_args()
    dataset = args.dataset
    segments_number = args.segments_number
    type_ = args.mode

    train, dev, test = prepare_input(dataset=dataset, segments_number=segments_number, text_segments=True, n_jobs=1)

    EF = fake_flow('{}_{}'.format(dataset, segments_number))
    EF.prepare_input(train, dev, test)

    if args.search == 0:
        EF.run_model(type_)
    else:
        EF.run_hyperopt_search(args.search)