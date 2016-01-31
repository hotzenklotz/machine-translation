from collections import defaultdict
from utils import nested_defaultdict, tokenize, iterate_nested_dict, HashableDict
from math import factorial
from sentence_pair import SentencePair


class IBMModel3:
    MIN_PROB = 1.0e-08

    def __init__(self, translations, alignments, sentence_pairs):

        self.sentence_pairs = sentence_pairs
        self.translations = translations
        self.alignments = alignments

        self.distortions = nested_defaultdict(4, int)
        self.fertilities = nested_defaultdict(2, lambda: 0.1)
        self.p1 = 0


    def train(self, num_iterations):

        # While not converged
        for step in range(0, num_iterations):
            print "Converging-Step ", step

            count_t = nested_defaultdict(2, lambda: IBMModel3.MIN_PROB)
            count_f = nested_defaultdict(2, lambda: IBMModel3.MIN_PROB)
            count_d = nested_defaultdict(4, lambda: IBMModel3.MIN_PROB)
            count_p0 = 0.0
            count_p1 = 0.0

            total_t = defaultdict(lambda: IBMModel3.MIN_PROB)
            total_f = defaultdict(lambda: IBMModel3.MIN_PROB)
            total_d = nested_defaultdict(3, lambda: IBMModel3.MIN_PROB)

            for sentence_pair in self.sentence_pairs:

                e_tokens = sentence_pair.e_tokens
                f_tokens = sentence_pair.f_tokens

                le = len(e_tokens)
                lf = len(f_tokens) - 1

                null_insertions = 0

                A, best_alignment = self.sample(sentence_pair)

                sentence_pair.alignment = best_alignment.alignment

                # E step (a): Compute normalization factors to weigh counts
                c_total = 0
                for a in A:
                    c_total += self.prob_t_a_given_s(a)

                # E step (b): Collect counts
                for a in A:
                    c = self.prob_t_a_given_s(a) / c_total

                    for j in range(1, lf):
                        aj = a.alignment[j] # shorthand, equivalent to i

                        en_word = e_tokens[j]
                        fr_word = f_tokens[aj - 1]

                        # lexical translations
                        count_t[en_word][fr_word] += c
                        total_t[fr_word] += c

                        # distortions
                        count_d[j][aj][le][lf] += c
                        total_d[aj][le][lf] += c

                        if aj == 0:  # Null insertion
                            null_insertions += 1

                    # Count null insertions
                    count_p1 += null_insertions * c
                    count_p0 += (le - 2 * null_insertions) * c

                    # Count fertilities
                    for i, fr_word in enumerate(f_tokens, 1):  # TODO NULL OTKEN OR NOT?
                        fertility = 0
                        for j in range(1, le):
                            if i == aj:
                                fertility += 1

                        count_f[fertility][fr_word] += c
                        total_f[fr_word] += c

            translations = nested_defaultdict(2, int)
            distortions = nested_defaultdict(4, int)
            fertilities = nested_defaultdict(2, int)

            # M Step :
            # Estimate probability distribution
            for ((en_word, fr_word), count_t) in iterate_nested_dict(count_t):
                translations[en_word][fr_word] = count_t / total_f[fr_word]

            # Estimate distortions
            for ((j, i, le, fe), count_d) in iterate_nested_dict(count_d):
                distortions[j][i][le][lf] = count_d / total_d[i][le][lf]

            # Estimate the fertility, n(Fertility | input word)
            for ((fertility, fr_word), count_f) in iterate_nested_dict(count_f):
                fertilities[fertility][fr_word] = count_f / total_f[fr_word]

            # Estimate the probability of null insertion
            p1 = count_p1 / (count_p1 + count_p0)
            # Clip p1 if it is too large, because p0 = 1 - p1 should not be
            # smaller than MIN_PROB
            self.p1 = min(p1, 1 - IBMModel3.MIN_PROB)

            self.distortions = distortions
            self.fertilities = fertilities
            self.translations = translations

    def sample(self, sentence_pair):
        A = set()

        lf = len(sentence_pair.f_tokens)
        le = len(sentence_pair.e_tokens)

        def find_best_alignment(j_pegged=None, i_pegged=0):
            new_alignment = SentencePair(sentence_pair)

            # find best alignment according to Model 2
            for j, fr_word in enumerate(sentence_pair.f_tokens, 1):  # TODO e + 1?
                if j == j_pegged:
                    best_i = i_pegged

                else:
                    best_i = 0
                    max_prob = 0

                    for i, en_word in enumerate(sentence_pair.e_tokens):
                        prob = self.translations[en_word][fr_word] * self.alignments[i][j][lf - 1][le - 1]
                        if prob >= max_prob:
                            max_prob = prob
                            best_i = i

                new_alignment.alignment[j] = best_i
                new_alignment.fertility[best_i].append(j)

            return new_alignment

        original_alignment = find_best_alignment()
        new_alignment = self.hillclimb(original_alignment)
        neighbor_alignment = self.neighboring(new_alignment)
        A.update(neighbor_alignment)

        best_alignment = new_alignment

        for j in range(1, le + 1):
            for i in range(0, lf + 1):

                # best IBM2 alignment
                original_alignment = find_best_alignment(j, i)
                new_alignment = self.hillclimb(original_alignment, j)
                neighbor_alignment = self.neighboring(new_alignment, j)
                A.update(neighbor_alignment)

                if new_alignment.alignment_prob > best_alignment.alignment_prob:
                    best_alignment = new_alignment

        return A, best_alignment

    def hillclimb(self, sentence_pair, j_pegged=None):
        max_prob = self.prob_t_a_given_s(sentence_pair)

        while True:
            old_alignment = sentence_pair
            for neighbor_alignment in self.neighboring(sentence_pair, j_pegged):
                neighbor_probability = self.prob_t_a_given_s(neighbor_alignment)

                if neighbor_probability > max_prob:
                    sentence_pair = neighbor_alignment
                    max_prob = neighbor_probability

            if sentence_pair == old_alignment:
                # Until there are no better alignments
                break

        sentence_pair.alignment_prob = max_prob
        return sentence_pair

    def prob_t_a_given_s(self, sentence_pair):

        lf = len(sentence_pair.f_tokens) - 1  # exclude NULL
        le = len(sentence_pair.e_tokens) - 1
        p1 = self.p1
        p0 = 1 - p1

        probability = 1.0
        MIN_PROB = IBMModel3.MIN_PROB

        # Combine NULL insert ion probability
        null_fertility = len(sentence_pair.fertility[0])
        probability *= (pow(p1, null_fertility) *
                        pow(p0, le - 2 * null_fertility))
        if probability < MIN_PROB:
            return MIN_PROB

        # Compute combination (m - null_fertility) choose null_fertility
        for i in range(1, null_fertility + 1):
            probability *= (le - null_fertility - i + 1) / i
            if probability < MIN_PROB:
                return MIN_PROB

        # Combine fertility probabilities
        for i in range(1, lf + 1):
            fertility = len(sentence_pair.fertility[i])
            probability *= (factorial(fertility) *
                            self.fertilities[fertility][sentence_pair.f_tokens[i]])
            if probability < MIN_PROB:
                return MIN_PROB

        # Combine lexical and distortion probabilities
        for j in range(1, le + 1):
            e = sentence_pair.e_tokens[j]
            i = sentence_pair.alignment[j]
            f = sentence_pair.f_tokens[i - 1] # alignments are one-indexed, tokens zero-indexed

            probability *= (self.translations[e][f] * self.distortions[j][i][lf][le])
            if probability < MIN_PROB:
                return MIN_PROB

        return probability

    def neighboring(self, sentence_pair, j_pegged=None):
        N = set()
        le = len(sentence_pair.e_tokens)
        lf = len(sentence_pair.f_tokens)

        for j in range(1, le):
            # moves
            if j == j_pegged:
                continue

            for i in range(0, lf):
                old_i = sentence_pair.alignment[j]

                new_align = SentencePair(sentence_pair) # create a copy
                new_align.alignment[j] = i
                new_align.fertility[i].append(j)
                new_align.fertility[old_i].remove(j)

                N.add(new_align)

        for j1 in range(1, le):
            # swaps
            if j1 == j_pegged:
                continue

            for j2 in range(1, le):
                if j2 == j_pegged or j2 == j1:
                    continue

                new_align = SentencePair(sentence_pair)
                i1 = sentence_pair.alignment[j1]
                i2 = sentence_pair.alignment[j2]


                # swap a_(j1), a_(j2)
                new_align.alignment[j1] = i2
                new_align.alignment[j2] = i1

                new_align.fertility[i2].remove(j2)
                new_align.fertility[i1].remove(j1)
                new_align.fertility[i2].append(j1)
                new_align.fertility[i1].append(j2)

                N.add(new_align)

        return N
