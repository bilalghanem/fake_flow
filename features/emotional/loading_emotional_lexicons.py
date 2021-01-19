import warnings, operator
warnings.filterwarnings("ignore")
import pandas as pd
from os.path import join


class emotional_lexicons:

    def __init__(self, path):
        self.lexicons_path = path

        # NRC, plutchik
        self.nrc = pd.read_csv(join(self.lexicons_path, 'nrc.txt'), sep='\t', names=["word", "emotion", "association"])
        self.nrc = self.nrc.pivot(index='word', columns='emotion', values='association').reset_index()
        del self.nrc['positive']
        del self.nrc['negative']

        # NRC (intensity), plutchik
        self.nrc_intensity = pd.read_csv(join(self.lexicons_path, 'NRC-AffectIntensity-Lexicon.txt'), sep='\t', names=["word", "score", "emotion"], skiprows=1)
        self.nrc_intensity = self.nrc_intensity.pivot(index='word', columns='emotion', values='score').reset_index()
        self.nrc_intensity.fillna(value=0, inplace=True)



    def frequency(self, sentence):
        words = []
        for word in sentence:
            try:
                result = self.nrc[self.nrc.word == str(word)].values.tolist()[0][1:]
            except:
                result = [0, 0, 0, 0, 0, 0, 0, 0]
                pass
            words.append(result)

        if len(words) == 0:
            words.append([0, 0, 0, 0, 0, 0, 0, 0])
        return words

    def intensity(self, sentence):
        words = []
        for word in sentence:
            try:
                result = self.nrc_intensity[self.nrc_intensity.word == str(word)].values.tolist()[0][1:]
            except:
                result = [0, 0, 0, 0]
                pass
            words.append(result)
        if len(words) == 0:
            words.append([0, 0, 0, 0])
        return words
