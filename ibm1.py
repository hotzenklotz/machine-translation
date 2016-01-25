# -*- coding: utf-8 -*-
from utils import tokenize, nested_defaultdict, iterate_nested_dict
from collections import defaultdict


class IBMModel1():
    ''' IBM1 Algorithm

    initialize t(e|f) uniformly
    do until convergence
       set count(e|f) to 0 for all e,f
       set total(f) to 0 for all f

       for all sentence pairs (e_s,f_s)

         for all words e in e_s
            set total_s(e) = 0

            for all words f in f_s
              total_s(e) += t(e|f)

         for all words e in e_s
          for all words f in f_s
             count(e|f) += t(e|f) / total_s(e)
             total(f)   += t(e|f) / total_s(e)

       for all f
         for all e
           t(e|f) = count(e|f) / total(f)
    '''

    def __init__(self, sentence_pairs):

        self.sentence_pairs = sentence_pairs
        self.translations = None

    def train(self, num_iterations=10):
        translations = self.translations

        # Uniformly init translations
        for e_s, f_s in self.sentence_pairs:
            e_tokens = tokenize(e_s)  # target language words
            f_tokens = tokenize(f_s)  # source language words

            uniform_prob = 1.0 / (len(e_tokens) + len(f_tokens))
            translations = nested_defaultdict(2, lambda: uniform_prob)

        # while not converged
        for step in range(0, num_iterations):
            print "Converging-Step: ", step

            # Initialization
            count = nested_defaultdict(2, int)
            total = defaultdict(int)
            total_s = defaultdict(int)

            for e_s, f_s in self.sentence_pairs:

                e_tokens = tokenize(e_s)
                f_tokens = [None] + tokenize(f_s)

                # E step (a): Compute normalization factors to weigh counts
                for e in e_tokens:
                    total_s[e] = 0.0

                    for f in f_tokens:
                        total_s[e] += translations[e][f]

                # E step (b): Collect counts
                for e in e_tokens:
                    for f in f_tokens:
                        count[e][f] += translations[e][f] / total_s[e]
                        total[f] += translations[e][f] / total_s[e]

            # M step: Update translations with maximum likelihood estimate
            for ((e, f), count_e_f) in iterate_nested_dict(count):
                translations[e][f] = count_e_f / total[f]

            self.translations = translations

        return translations
