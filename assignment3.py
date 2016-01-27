import time
from copy import deepcopy
from ibm1 import IBMModel1
from ibm2 import IBMModel2
from news_corpus import NewsCorpus
from language_model import LanguageModel
from word_alignment import grow_diag_final
from utils import matrix, tokenize


def translate(sentence, ibm_model, language_model):
    print ibm_model.predict(sentence)
    print ibm_model.predict_with_language_model(sentence, language_model)

def calc_alignments(english, german, ibm_model_e_to_f, ibm_model_f_to_e):

    e_tokens = tokenize(english)
    f_tokens = tokenize(german)

    alignment_e_to_f = ibm_model_e_to_f.align(english, german)
    alignment_f_to_e = ibm_model_f_to_e.align(german, english)
    alignment = grow_diag_final(len(e_tokens), len(f_tokens), alignment_e_to_f, alignment_f_to_e)
    print len(alignment_e_to_f), len(alignment_f_to_e), len(alignment)

    print alignment
    print matrix(e_tokens, f_tokens, alignment)

    return alignment

if __name__ == '__main__':

    start_time = time.time()

    test_pairs = [
        ("the house", "das haus"),
        ("the book", "das buch"),
        ("a book", "ein buch")
    ]

    print "Starting Data import..."
    # (english, german) = sentence_pairs
    # en_ger_sentence_pairs = NewsCorpus().get_test_sentence_pairs()
    en_ger_sentence_pairs = NewsCorpus().get_sentence_pairs()
    # en_ger_sentence_pairs = test_pairs

    ger_en_sentence_pairs = ([(german, english) for (english, german) in en_ger_sentence_pairs])

    print "Starting Language Model Training..."
    # Train a German language model
    language_model = LanguageModel()
    language_model.train([german for (english, german) in en_ger_sentence_pairs])

    print "Starting IBM Model 1 Training..."
    ibm1_en_to_ger = IBMModel1(en_ger_sentence_pairs)
    ibm1_en_to_ger.train(10)

    ibm1_ger_to_en = IBMModel1(ger_en_sentence_pairs)
    ibm1_ger_to_en.train(10)


    print "Starting IBM Model 2 Training..."
    ibm2_en_to_ger = IBMModel2(ibm1_en_to_ger.translations, en_ger_sentence_pairs)
    ibm2_en_to_ger.train(5)

    ibm2_ger_to_en = IBMModel2(ibm1_ger_to_en.translations, ger_en_sentence_pairs)
    ibm2_ger_to_en.train(5)

    print "Translating first 20 test sentences..."
    # Translate the English sentences into German
    for (english, german) in en_ger_sentence_pairs[0:20]:

        print "--------"
        print english


        translate(german, ibm2_en_to_ger, language_model)
        calc_alignments(english, german, ibm2_en_to_ger, ibm2_ger_to_en)


    print "Translation took %ss" % (time.time() - start_time)

