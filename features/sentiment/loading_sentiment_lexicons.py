import pandas as pd
from os.path import join


class sentiment_lexicons:
    def __init__(self, path=''):
        self.nrc = pd.read_csv(join(path, 'nrc.txt'), sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()
        self.positive = self.nrc[self.nrc['positive'] == 1]['word'].tolist()
        self.negative = self.nrc[self.nrc['negative'] == 1]['word'].tolist()

    def score(self, sentence):
        words = []
        for word in sentence:
            try:
                pos = 1 if word in self.positive else 0
                neg = 1 if word in self.negative else 0
                result = [pos, neg]
            except:
                result = [0, 0]
                pass
            words.append(result)

        if len(words) == 0:
            words.append([0, 0])
        return words

