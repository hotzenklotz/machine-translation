import time
from ibm1 import IBMModel1
from ibm2 import IBMModel2
from news_corpus import NewsCorpus
from language_model import LanguageModel
from word_alignment import grow_diag_final
from utils import matrix


def translate_sentence(sentence, ibm_model_e_to_f, ibm_model_f_to_e, language_model):

    translation = ibm_model_e_to_f.predict_with_language_model(sentence, language_model)

    alignment_e_to_f = ibm_model_e_to_f.align(sentence, translation)
    alignment_f_to_e = ibm_model_f_to_e.align(sentence, translation)
    alignment = grow_diag_final(len(sentence), len(translation), alignment_e_to_f, alignment_f_to_e)

    return translation, alignment

if __name__ == '__main__':

    start_time = time.time()

    test_pairs = [
        ("the house", "das haus"),
        ("the book", "das buch"),
        ("a book", "ein buch")
    ]

    print "Starting Data import..."
    # (english, german) = sentence_pairs
    sentence_pairs = NewsCorpus().get_test_sentence_pairs()
    # sentence_pairs = NewsCorpus().get_sentence_pairs()
    # sentence_pairs = test_pairs

    print "Starting Language Model Training..."
    # Train a German language model
    language_model = LanguageModel()
    language_model.train([german for (english, german) in sentence_pairs])

    print "Starting IBM Model 1 Training..."
    ibm1 = IBMModel1(sentence_pairs)
    ibm1.train(10)

    print "Starting IBM Model 2 Training..."
    ibm2_en_to_ger = IBMModel2(ibm1.probabilities, sentence_pairs)
    ibm2_en_to_ger.train(5)

    ibm2_ger_to_en = IBMModel2(ibm1.probabilities, ([(german, english) for (english, german) in sentence_pairs]))
    ibm2_ger_to_en.train(5)

    print "Translating first 20 test sentences..."
    # Translate the English sentences into German
    for (english, german) in sentence_pairs[0:20]:
        translation, alignment = translate_sentence(english, ibm2_en_to_ger, ibm2_ger_to_en, language_model)

        print "--------"
        print english
        print translation
        print alignment
        print matrix(english, translation, alignment)

    print "Translation took %ss" % (time.time() - start_time)

