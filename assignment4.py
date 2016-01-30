import time
from ibm1 import IBMModel1
from ibm2 import IBMModel2
# from model3 import IBMModel3
from ibm3 import IBMModel3
from news_corpus import NewsCorpus
import dill

if __name__ == '__main__':

    start_time = time.time()

    print "Starting Data import..."
    # (english, german) = sentence_pairs
    en_ger_sentence_pairs = NewsCorpus().get_test_sentence_pairs()
    # en_ger_sentence_pairs = NewsCorpus().get_sentence_pairs()
    # en_ger_sentence_pairs = test_pairs

    ger_en_sentence_pairs = ([(german, english) for (english, german) in en_ger_sentence_pairs])

    print "Starting IBM Model 1 Training..."
    # ibm1 = IBMModel1(en_ger_sentence_pairs)
    # ibm1.train(10)


    print "Starting IBM Model 2 Training..."
    #ibm2 = IBMModel2(ibm1.translations, en_ger_sentence_pairs)
    #ibm2.train(5)

    #dill.dump(ibm2, open("ibm2.p", "wb"))
    ibm2 = dill.load(open("ibm2.p", "rb"))

    print "Starting IBM Model 3 Training..."
    ibm3 = IBMModel3(ibm2.translations, ibm2.alignments, en_ger_sentence_pairs)
    ibm3.train(5)

    print "Translating first 20 test sentences..."
    # Translate the English sentences into German
    for (english, german) in en_ger_sentence_pairs[0:20]:

        print "--------"
        print english, german


    print "Translation took %ss" % (time.time() - start_time)

