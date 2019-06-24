import codecs
import re
import numpy as np
from nltk.tokenize import sent_tokenize
from nltk import word_tokenize, pos_tag
from nltk.stem.snowball import SnowballStemmer

class Corpus(object):

    def __init__(self, tokens_size):
        self.tokens_size = tokens_size
        self.total_tokens = 0
        self.avg_w_len = 0
        self.total_sent = 0
        self.avg_sent_len = 0
        self.set_words = set()
        self.TTRs = list()
        self.freq_classes = list()
        self.avg_verb_sent = 0
        self.avg_noun_sent = 0
        self.avg_adj_sent = 0
        self.lexical_density = 0
        self.tokens = list()
        self.sentence_tokens = list()
        
    def lexicaldensity(self, adv, verb, adj, noun, denominator):

        nominator = adv+verb+adj+noun

        return nominator/denominator


    def tagextractor(self, sentences, t):

        avgs = []
        total = 0
        for x in sentences:
            taggati = pos_tag(x)
            verbs = 0
            for y in taggati:
                if y[1].startswith(t):
                    verbs += 1
                    total += 1
            avgs.append((verbs / len(x)))

        return (avgs, total)

    def classefreq(self, tokens, freq):
        s = set(tokens)
        scores = dict()

        for x in s:
            if tokens.count(x) == freq:
                scores[x] = tokens.count(x)

        return scores

    def ttr(self, tokens):
        numerator = len(set(tokens))
        denominator = len(tokens)
        ttrscore = numerator / (np.sqrt(denominator))
        return ttrscore

    def cleaner(self,text):
        text = re.sub("[\.\-,;\<\>:\"\"\(\)\!\?]+", " ", text)
        return text

    def build_corpus(self, text):

        textTokens = self.cleaner(text)
        self.tokens = word_tokenize(text)
        self.tokens = self.tokens[:self.tokens_size]
        tottokens = len(self.tokens)
        avgwordlen = [len(x) for x in self.tokens]
        avgwordlen = np.mean(avgwordlen)
        self.sentence_tokens = sent_tokenize(text)
        totsent = len(self.sentence_tokens)
        avgsent = [len(x[:-1]) for x in self.sentence_tokens]
        avgsent = np.mean(avgsent)
        uniquewords = len(set(self.tokens))
        ttrscores = dict()

        for x in range(1000,6000,1000):

            score = self.ttr(self.tokens[:x])
            ttrscores[x] = score

        classefreqscores = []

        for x in range(3,10,3):
            classefreqscores.append(self.classefreq(self.tokens[:self.tokens_size], x))

        avgverbs = self.tagextractor(self.tokens[:self.tokens_size], "V")[0]
        avgnouns = self.tagextractor(self.tokens[:self.tokens_size], "N")[0]
        avgadj = self.tagextractor(self.tokens[:self.tokens_size], "J")[0]

        totverbs = self.tagextractor(self.tokens[:self.tokens_size], "V")[1]
        totnouns = self.tagextractor(self.tokens[:self.tokens_size], "N")[1]
        totadj = self.tagextractor(self.tokens[:self.tokens_size], "J")[1]
        totadv = self.tagextractor(self.tokens[:self.tokens_size], "RB")[1]

        lexdensity = self.lexicaldensity(totverbs,totnouns,totadj,totadv, tottokens)

        self.tokens_size = tottokens
        self.avg_w_len = avgwordlen
        self.total_sent = totsent
        self.avg_sent_len = avgsent
        self.set_words = uniquewords
        self.TTRs = ttrscores
        self.freq_classes = classefreqscores
        self.avg_verb_sent = avgverbs
        self.avg_noun_sent = avgnouns
        self.avg_adj_sent = avgadj
        self.lexical_density = lexdensity
        
        return None

