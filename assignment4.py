# -*- coding: utf-8 -*-

import time
import dill
from ibm1 import IBMModel1
from ibm2 import IBMModel2
from ibm3 import IBMModel3
from news_corpus import NewsCorpus
from sentence_pair import SentencePair

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
        ["das buch is ja klein", "the book is small"],
        ["das haus", "the house"],
        ["das buch", "the book"],
        ["ein buch", "a book"],
        ["ich fasse das buch zusammen", "i summarize the book"],
        ["fasse zusammen", "summarize"]
    ]

    sentence_pairs = [SentencePair(english, german) for (english, german) in en_ger_sentence_pairs]


    print "Starting IBM Model 1 Training..."
    ibm1 = IBMModel1(en_ger_sentence_pairs)
    ibm1.train(10)


    print "Starting IBM Model 2 Training..."
    ibm2 = IBMModel2(ibm1.translations, en_ger_sentence_pairs)
    ibm2.train(5)

    dill.dump(ibm2, open("ibm2.p", "wb"))
    # ibm2 = dill.load(open("ibm2.p", "rb"))

    print "Starting IBM Model 3 Training..."
    ibm3 = IBMModel3(ibm2.translations, ibm2.alignments, sentence_pairs)
    ibm3.train(5)

    dill.dump(ibm3, open("ibm3.p", "wb"))


    print ibm3.alignments, ibm3.fertilities

    print "Translating first 20 test sentences..."
    # Translate the English sentences into German
    for (english, german) in en_ger_sentence_pairs[0:20]:

        print "--------"
        print english, german


    print "Translation took %ss" % (time.time() - start_time)

