# -*- coding: utf-8 -*-
from utils import tokenize
from collections import defaultdict
from copy import deepcopy

class SentencePair(object):

    def __init__(self, english_sentence, german_sentence=""):

        # overload constructor
        if type(english_sentence) is SentencePair:
            self.copyFromObj(english_sentence)
            return


        self.english_sentence = english_sentence
        self.german_sentence = german_sentence

        self.e_tokens = tokenize(english_sentence)
        self.f_tokens = [None] + tokenize(german_sentence)

        self.alignment = defaultdict(int)
        self.alignment_prob = 0.0

        self.fertility = defaultdict(int)

    def copyFromObj(self, obj):

        self.english_sentence = obj.english_sentence
        self.german_sentence = obj.german_sentence

        self.e_tokens = obj.e_tokens
        self.f_tokens = obj.f_tokens

        self.alignment = deepcopy(obj.alignment)
        self.alignment_prob = float(obj.alignment_prob)
        self.fertility = deepcopy(obj.fertility)

