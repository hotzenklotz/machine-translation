# -*- coding: utf-8 -*-

import time
import dill
from ibm1 import IBMModel1
from ibm2 import IBMModel2
from ibm3 import IBMModel3
from news_corpus import NewsCorpus
from sentence_pair import SentencePair
from utils import matrix

if __name__ == '__main__':

    start_time = time.time()

    print "Starting Data import..."
    # en_ger_sentence_pairs = NewsCorpus().get_test_sentence_pairs()
    # en_ger_sentence_pairs = NewsCorpus().get_sentence_pairs()
    # en_ger_sentence_pairs = NewsCorpus().get_sentence_pairs()
    # en_ger_sentence_pairs = test_pairs

    en_ger_sentence_pairs = [
        ["klein ist das haus", "the house is small"],
        ["das haus war ja groß", "the house was big"],
        ["das buch ist ja klein", "the book is small"],
        ["das haus", "the house"],
        ["das buch", "the book"],
        ["ein buch", "a book"],
        ["ich fasse das buch zusammen", "i summarize the book"],
        ["fasse zusammen", "summarize"]
    ]

    sentence_pairs = [SentencePair(english, german) for (german, english) in en_ger_sentence_pairs]

    print "Starting IBM Model 1 Training..."
    ibm1 = IBMModel1(en_ger_sentence_pairs)
    ibm1.train(10)


    print "Starting IBM Model 2 Training..."
    ibm2 = IBMModel2(ibm1.translations, en_ger_sentence_pairs)
    ibm2.train(5)

    # dill.dump(ibm2, open("ibm2.p", "wb"))
    # ibm2 = dill.load(open("ibm2.p", "rb"))

    print "Starting IBM Model 3 Training..."
    ibm3 = IBMModel3(ibm2.translations, ibm2.alignments, sentence_pairs)
    ibm3.train(5)

    # Save a snapshot
    dill.dump(ibm3, open("ibm3.p", "wb"))
    # ibm3 = dill.load(open("ibm3.p", "rb"))


    for s in sentence_pairs:
        alignment = [(key, value) for (key, value) in s.alignment.items()]
        print s.alignment

        print matrix(s.e_tokens, s.f_tokens[1:], alignment)

    print "Processing took %ss" % (time.time() - start_time)
