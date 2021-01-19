from os.path import join
import pandas as pd
import ast

class imageability_class:

    def __init__(self, path=''):
        self.reader = pd.read_csv(join(path, 'imageability.predictions'), sep='\t', names=["word", "class", "association"])
        association = self.reader['association'].tolist()
        association = pd.DataFrame([ast.literal_eval(item) for item in association])
        association.rename(columns={'A': 'imageability_prob', 'C': 'abstraction_prob'}, inplace=True)
        del self.reader['association']
        self.reader = pd.concat([self.reader, association], axis=1)
        self.img = self.reader[(self.reader['imageability_prob'] > 0.9)]['word'].tolist()
        self.abs = self.reader[(self.reader['abstraction_prob'] > 0.8)]['word'].tolist()

    def score(self, sentence):
        words = sentence
        results = []
        for word in words:
            img = 1 if word in self.img else 0
            abs = 1 if word in self.abs else 0
            results.append([img, abs])
        if len(results) == 0:
            results.append([0, 0])
        return results
