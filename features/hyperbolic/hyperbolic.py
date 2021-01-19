from os.path import join
import pandas as pd

class hyperbolic_class:

    def __init__(self, path=''):
        self.reader = pd.read_csv(join(path, 'hyperbolic'), sep='\n', names=["word"])
        self.hyper = self.reader['word'].tolist()

    def score(self, sentence):
        words = sentence
        results = []
        for word in words:
            hyper = 1 if word in self.hyper else 0
            results.append([hyper])

        if len(results) == 0:
            results.append([0])
        return results
