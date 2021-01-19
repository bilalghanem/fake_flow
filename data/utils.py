import numpy as np
from sklearn.utils import shuffle

np.random.seed(0)


def split(data, data_features, return_features, dev_ratio=0.3):
    train = data[data.type == 'training']
    train_features = data_features[train.index, :, :] if return_features else []
    train = shuffle(train)
    train = train.reset_index(drop=True).reset_index()
    del train['id']
    train = train.rename(columns={'index': 'id'})

    test = data[data.type == 'test']
    test_features = data_features[test.index, :, :] if return_features else []
    test = shuffle(test)
    test = test.reset_index(drop=True).reset_index()
    del test['id']
    test = test.rename(columns={'index': 'id'})

    self_train = {}
    self_dev = {}
    self_test = {}

    msk_dev = np.random.rand(len(train)) < dev_ratio
    self_dev['text'] = train['content'][msk_dev]
    self_dev['features'] = train_features[msk_dev, :, :] if return_features else []
    self_dev['label'] = train['label'][msk_dev]
    train = train[~msk_dev]
    self_train['text'] = train['content']
    self_train['features'] = train_features[~msk_dev, :, :] if return_features else []
    self_train['label'] = train['label']

    self_test['text'] = test['content']
    self_test['features'] = test_features if return_features else []
    self_test['label'] = test['label']
    return self_train, self_dev, self_test
