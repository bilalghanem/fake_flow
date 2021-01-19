import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from features.building_features import manual_features, segmentation_text, clean_regex
from data.utils import split

np.random.seed(0)


def prepare_input(dataset='MultiSourceFake', segments_number=10, n_jobs=-1, emo_rep='frequency', return_features=True,
                  text_segments=False, clean_text=True):

    content = pd.read_csv('./data/{}/sample.csv'.format(dataset))
    content_features = []
    """Extract features, segment text, clean it."""
    if return_features:
        content_features = manual_features(n_jobs=n_jobs, path='./features', model_name=dataset,
                                           segments_number=segments_number, emo_rep=emo_rep).transform(content['content'])

    """In segmentation we already clean the text to keep the DOTS (.) only."""
    if text_segments:
        content['content'] = segmentation_text(segments_number=segments_number).transform(content['content'])
    elif clean_text:
        content['content'] = content['content'].map(lambda text: clean_regex(text, keep_dot=True))

    train, dev, test = split(content, content_features, return_features)
    return train, dev, test



if __name__ == '__main__':
    train, dev, test = prepare_input(dataset='MultiSourceFake', segments_number=10, n_jobs=-1, text_segments=True)