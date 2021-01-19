from os.path import join
import pandas as pd
import re
from features.morality.morality_readDict import readDict

class MORALITY_class:

    def __init__(self, path=''):
        # Morality
        self.mor = readDict(join(path, 'moral foundations dictionary-edited.dic'))
        self.mor = pd.DataFrame(self.mor, columns=['word', 'category'])
        self.mor['word'] = self.mor['word'].map(lambda x: re.sub(r'[*]', '', x))
        self.mor['value'] = 1
        self.mor = pd.pivot_table(self.mor, index='word', columns=['category'],
                                   values='value', fill_value=0).reset_index().reindex(['word', 'HarmVirtue', 'HarmVice', 'FairnessVirtue', 'FairnessVice', 'IngroupVirtue', 'IngroupVice', 'AuthorityVirtue', 'AuthorityVice', 'PurityVirtue', 'PurityVice', 'MoralityGeneral'], axis=1)

        self.HarmVirtue = self.mor[self.mor['HarmVirtue'] == 1]['word'].tolist()
        self.HarmVice = self.mor[self.mor['HarmVice'] == 1]['word'].tolist()
        self.FairnessVirtue = self.mor[self.mor['FairnessVirtue'] == 1]['word'].tolist()
        self.FairnessVice = self.mor[self.mor['FairnessVice'] == 1]['word'].tolist()
        self.IngroupVirtue = self.mor[self.mor['IngroupVirtue'] == 1]['word'].tolist()
        self.IngroupVice = self.mor[self.mor['IngroupVice'] == 1]['word'].tolist()
        self.AuthorityVirtue = self.mor[self.mor['AuthorityVirtue'] == 1]['word'].tolist()
        self.AuthorityVice = self.mor[self.mor['AuthorityVice'] == 1]['word'].tolist()
        self.PurityVirtue = self.mor[self.mor['PurityVirtue'] == 1]['word'].tolist()
        self.PurityVice = self.mor[self.mor['PurityVice'] == 1]['word'].tolist()
        self.MoralityGeneral = self.mor[self.mor['MoralityGeneral'] == 1]['word'].tolist()

    def score(self, sentence):
        words = sentence
        results = []
        for word in words:
            HarmVirtue = 1 if word in self.HarmVirtue else 0
            HarmVice = 1 if word in self.HarmVice else 0
            FairnessVirtue = 1 if word in self.FairnessVirtue else 0
            FairnessVice = 1 if word in self.FairnessVice else 0
            IngroupVirtue = 1 if word in self.IngroupVirtue else 0
            IngroupVice = 1 if word in self.IngroupVice else 0
            AuthorityVirtue = 1 if word in self.AuthorityVirtue else 0
            AuthorityVice = 1 if word in self.AuthorityVice else 0
            PurityVirtue = 1 if word in self.PurityVirtue else 0
            PurityVice = 1 if word in self.PurityVice else 0
            MoralityGeneral = 1 if word in self.MoralityGeneral else 0
            results.append([HarmVirtue, HarmVice, FairnessVirtue, FairnessVice, IngroupVirtue, IngroupVice,
                            AuthorityVirtue, AuthorityVice, PurityVirtue, PurityVice, MoralityGeneral])
        if len(results) == 0:
            results.append([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        return results
