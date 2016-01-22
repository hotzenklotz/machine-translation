# -*- coding: utf-8 -*-
from utils import tokenize, nested_defaultdict, iterate_nested_dict
from collections import defaultdict


class IBMModel2():


    ''' IBM 2 Algorithm
    carry over t(e|f) from Model 1
    initialize a(i|j,le,lf) = 1/(lf+1) for all i,j,le,lf

    while not converged do

      // initialize
      count(e|f) = 0 for all e,f
      total(f) = 0 for all f
      counta(i|j,le,lf) = 0 for all i,j,le,lf
      totala(j,le,lf) = 0 for all j,le,lf

      for all sentence pairs (e,f) do
        le = length(e), lf = length(f)

        // compute normalization
        for j =1.. le do// all word positions in e
          s-total(ej) = 0

          for i =0.. lf do// all word positions in f
            s-total(ej) += t(ej|fi) ∗ a(i|j,le,lf)
          end for
        end for

        // collect counts
        for j =1.. le do // all word positions in e
          for i =0.. lf do // all word positions in f
            c = t(ej|fi) ∗ a(i|j,le,lf) / s-total(ej)
            count(ej|fi) += c
            total( fi ) += c
            counta(i|j,le,lf) += c
            totala(j,le,lf) += c
          end for
        end for

      end for

      // estimate probabilities
      t(e|f) = 0 for all e,f
      a(i|j,le,lf) = 0 for all i,j,le,lf

      for all e,f do
        t(e|f) = count(e|f) / total(f)
      end for
      for all i,j,le,lf do
        a(i|j,le,lf) = counta(i|j,le,lf) / totala(j,le,lf)
      end for
    end while
    '''

    def __init__(self, ibm1_model, sentence_pairs):
        self.sentence_pairs = sentence_pairs

        self.probabilities = ibm1_model
        self.alignments = None

    def train(self, num_iteration=10):
        probabilities = self.probabilities
        alignments = self.alignments

        for e_s, f_s in self.sentence_pairs:
            f_tokens = tokenize(f_s)
            lf = len(f_tokens)

            alignments = nested_defaultdict(4, lambda: 1.0 / (lf + 1.0))

        # while not converged
        for step in range(0, num_iteration):
            print "Converging-Step: ", step

            # initialize
            stotal = defaultdict(int)
            count = nested_defaultdict(2, int)
            total = defaultdict(int)
            count_a = nested_defaultdict(4, int)
            total_a = nested_defaultdict(3, int)

            for e_s, f_s in self.sentence_pairs:

                e_tokens = tokenize(e_s)  # target language words
                f_tokens = [None] + tokenize(f_s)  # source language words

                le = len(e_tokens)
                lf = len(f_tokens) - 1

                # E step (a): Compute normalization factors to weigh counts
                for j, e in enumerate(e_tokens):
                    e = e_tokens[j]
                    stotal[e] = 0

                    for i, f in enumerate(f_tokens):
                        stotal[e] += probabilities[e][f] * alignments[i][j][le][lf]

                # E step (b): Collect counts
                for j, e in enumerate(e_tokens):
                    for i, f in enumerate(f_tokens):
                        c = probabilities[e][f] * alignments[i][j][le][lf] / stotal[e]
                        count[e][f] += c
                        total[f] += c
                        count_a[i][j][le][lf] += c
                        total_a[j][le][lf] += c

            # M step: Update probabilities with maximum likelihood estimate
            for ((e, f), count_e_f) in iterate_nested_dict(count):
                probabilities[e][f] = count_e_f / total[f]

            for ((i, j, le, lf), counta_i_j) in iterate_nested_dict(count_a):
                alignments[i][j][le][lf] = counta_i_j / total_a[j][le][lf]

            self.probabilities = probabilities
            self.alignments = alignments

        return probabilities, alignments

    def predict_with_language_model(self, sentence, language_model):
        # Lexical Translation
        # argmax_e p(e|f) = argmax_e p(f|e) * p(e)

        tokens = tokenize(sentence)
        out = []

        for i, source_word in enumerate(tokens):

            best_translation = None
            best_prob = 0
            previousIndex = i - 1
            previous_target_word = out[previousIndex] if i > 0 and previousIndex < len(out) else None

            for ((target_word, ), ibm_prob) in iterate_nested_dict(self.probabilities[source_word]):
                combined_prob = ibm_prob * language_model[previous_target_word][target_word]
                if combined_prob >= best_prob:
                    best_prob = combined_prob
                    best_translation = target_word

            if not best_translation == None:
                out.append(best_translation)

        return " ".join(out)

    def align(self, src_sentence, target_sentence):
        return
