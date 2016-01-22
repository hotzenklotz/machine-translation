from collections import defaultdict
from utils import tokenize, nested_defaultdict


# Train a language model using Bi-Grams
def train_language_model(corpus):
    # Default to a very low probability higher than zero
    model = nested_defaultdict(2, lambda: 0.001)
    bigrams = nested_defaultdict(2, int)

    for i, sentence in enumerate(corpus):

        if i % 100 == 0:
            print "Processed %s bigrams" % i

        # Generate all NGrams and count them
        for (w1, w2) in get_bigrams(sentence):
            bigrams[w1][w2] += 1.0

        # Calculate probabilities
        num_bigrams = sum([len(w1) for (_, w1) in bigrams.iteritems()])

        for w1, w2 in bigrams.iteritems():
            ###
            # Maximum Likelihood Estimation with Add-One Smoothing
            # p(w2|w1) = count(w1, w2) + 1 / ( sum_w (count(w1 , w)) + len(bigrams))
            ###

            num_relevant_bigrams = sum([count for (w2, count) in bigrams[w1].iteritems()])
            model[w1][w2] = bigrams[w1][w2] + 1.0 / float(num_relevant_bigrams + num_bigrams)

    # return DefaultDict{w1, w2 : Prob, ...}
    return model


def get_bigrams(sentence):
    return get_ngrams(sentence, 2)


def get_ngrams(sentence, ngram_order):
    # Include 'None' token, so that every sentence can start with a bigram
    tokens = [None] + tokenize(sentence)
    ngrams = []

    if len(tokens) < ngram_order:
        # raise Exception("Sentence too short for NGram language model of order %s" % ngram_order)
        print "Sentence too short for NGram language model of order %s" % ngram_order
        print sentence
        return ngrams

    for i in range(0, len(tokens) - ngram_order - 1):
        ngrams.append(tokens[i:i + ngram_order])

    # return List[List[w1, w2], ...]
    return ngrams
