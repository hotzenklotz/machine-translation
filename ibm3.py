from collections import defaultdict
from utils import nested_defaultdict, tokenize, iterate_nested_dict, HashableDict
from math import factorial


class IBMModel3:
    MIN_PROB = 1.0e-12

    def __init__(self, translations, alignments, sentence_pairs):

        self.sentence_pairs = sentence_pairs
        self.translations = translations
        self.alignments = alignments

        self.distortions = nested_defaultdict(4, int)
        self.fertilities = nested_defaultdict(2, lambda: 0.1)
        self.p1 = 0
        self.null_inserts = defaultdict(lambda: [1])

    def train(self, num_iterations):

        # While not converged
        for step in range(0, num_iterations):
            print "Converging-Step ", step

            count_t = nested_defaultdict(2, int)
            count_f = nested_defaultdict(2, int)
            count_d = nested_defaultdict(4, int)
            count_p0 = 0.0
            count_p1 = 0.0

            total_t = defaultdict(int)
            total_f = defaultdict(int)
            total_d = nested_defaultdict(3, int)

            for (e_s, f_s) in self.sentence_pairs:

                e_tokens = tokenize(e_s)
                f_tokens = [None] + tokenize(f_s)

                le = len(e_tokens)
                lf = len(f_tokens) - 1

                null_insertions = 0

                A, best_alignment = self.sample(e_tokens, f_tokens)

                # E step (a): Compute normalization factors to weigh counts
                c_total = 0
                for alignment in A:
                    c_total += self.prob_t_a_given_s(alignment, e_tokens, f_tokens)

                # E step (b): Collect counts
                for a in A:
                    c = self.prob_t_a_given_s(a, e_tokens, f_tokens) / c_total

                    for j in range(0, le + 1):
                        en_word = e_tokens[j]
                        fr_word = f_tokens[a[j]]

                        # lexical translations
                        count_t[en_word][fr_word] += c
                        total_t[fr_word] += c

                        # distortions
                        count_d[j][a[j]][le][lf] += c
                        total_d[a[j]][le][lf] += c

                        if a[j] == 0:  # Null insertion
                            null_insertions += 1

                    # Count null insertions
                    count_p1 += null_insertions * c
                    count_p0 += (le - 2 * null_insertions) * c

                    # Count fertilities
                    for i, fr_word in enumerate(f_tokens, 1):  # TODO NULL OTKEN OR NOT?
                        fertility = 0
                        for j in range(0, le):
                            if i == a[j]:
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

    def sample(self, e_tokens, f_tokens):
        A = set()

        lf = len(f_tokens)
        le = len(e_tokens)

        def find_best_alignment(j_pegged=None, i_pegged=0):
            alignment = defaultdict(int)

            # find best alignment according to Model 2
            for j, en_word in enumerate(e_tokens, 1):  # TODO e + 1?
                if j == j_pegged:
                    best_i = i_pegged

                else:
                    best_i = 0
                    max_prob = 0

                    for i, fr_word in enumerate(f_tokens):
                        prob = self.translations[en_word][fr_word] * self.alignments[i][j][lf - 1][le - 1]
                        if prob >= max_prob:
                            max_prob = prob
                            best_i = i

                alignment[j] = best_i

            return alignment, max_prob

        original_alignment, _ = find_best_alignment()
        new_alignment, _ = self.hillclimb(original_alignment, e_tokens, f_tokens)
        neighbor_alignment = self.neighboring(new_alignment, e_tokens, f_tokens)
        A.update(neighbor_alignment)

        best_alignment = new_alignment

        for i in range(0, lf + 1):
            for j in range(1, le + 1):

                # best IBM2 alignment
                original_alignment, best_prob = find_best_alignment(j, i)
                new_alignment, new_prob = self.hillclimb(best_alignment, e_tokens, f_tokens, j)
                neighbor_alignment = self.neighboring(new_alignment, e_tokens, f_tokens, j)
                A.update(neighbor_alignment)

                if best_prob > new_prob:
                    best_alignment = new_alignment

        return A, best_alignment

    def hillclimb(self, alignment, e_tokens, f_tokens, j_pegged=None):
        max_probability = self.prob_t_a_given_s(alignment, e_tokens, f_tokens)

        while True:
            old_alignment = alignment
            for neighbor_alignment in self.neighboring(alignment, e_tokens, f_tokens, j_pegged):
                neighbor_probability = self.prob_t_a_given_s(neighbor_alignment, e_tokens, f_tokens)

                if neighbor_probability > max_probability:
                    alignment = neighbor_alignment
                    max_probability = neighbor_probability

            if alignment == old_alignment:
                # Until there are no better alignments
                break

        # alignment.score = max_probability
        return alignment, max_probability

    def prob_t_a_given_s(self, alignment, e_tokens, f_tokens):

        lf = len(f_tokens) - 1  # exclude NULL
        le = len(e_tokens) - 1
        p1 = self.p1
        p0 = 1 - p1

        probability = 1.0
        MIN_PROB = IBMModel3.MIN_PROB

        # Combine NULL insert ion probability
        null_fertility = len(self.null_inserts[0])
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
            fertility = len(self.null_inserts[i])
            probability *= (factorial(fertility) *
                            self.fertilities[fertility][f_tokens[i]])
            if probability < MIN_PROB:
                return MIN_PROB

        # Combine lexical and distortion probabilities
        for j in range(1, le + 1):
            e = e_tokens[j]
            i = alignment[j]
            f = f_tokens[i]

            probability *= (self.translations[e][f] * self.distortions[j][i][lf][le])
            if probability < MIN_PROB:
                return MIN_PROB

        return probability

    def neighboring(self, alignments, e_tokens, f_tokens, j_pegged=None):
        N = set()
        for j in range(1, len(e_tokens) + 1):
            # moves
            if j == j_pegged:
                continue

            for i in range(0, len(f_tokens) + 1):
                new_align = HashableDict(alignments)
                new_align[j] = i

                N.add(new_align)

        for j1 in range(1, len(e_tokens) + 1):
            # swaps
            if j1 == j_pegged:
                continue

            for j2 in range(1, len(e_tokens) + 1):
                if j2 == j_pegged or j2 == j1:
                    continue

                new_align = HashableDict(alignments)

                # swap a_(j1), a_(j2)
                tmp = new_align[j1]
                new_align[j1] = new_align[j2]
                new_align[j2] = tmp

                N.add(new_align)

        return N
