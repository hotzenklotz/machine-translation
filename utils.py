import regex
from collections import defaultdict

# p{L} Letter Symbol (unicode)
# p{Nd} Decimal Number
TOKENIZER = regex.compile("[^\\p{L}\\p{Nd}]+")  # ("\s+")
REMOVE_PUNCTUATION = regex.compile("[^\w\s]")


def tokenize(text):
    text = text.strip().lower()
    text_without_punctuation = REMOVE_PUNCTUATION.sub("", text)

    # discard empty tokens
    return [token for token in TOKENIZER.split(text_without_punctuation) if token]


def nested_defaultdict(levels=1, final=int):
    return (defaultdict(final) if levels < 2 else
            defaultdict(lambda: nested_defaultdict(levels - 1, final)))


def iterate_nested_dict(dict, keys=()):
    if type(dict) == defaultdict:
        for key, value in dict.iteritems():
            for rv in iterate_nested_dict(value, keys + (key,)):
                yield rv
    else:
        yield (keys, dict)


def swap_keys(nested_dict):
    # Swap the keys of a nested object
    # a[i][j] = value --> b[j][i] = value
    inverse_dict = nested_defaultdict(2, int)

    for ((key1, key2), value) in iterate_nested_dict(nested_dict):
        inverse_dict[key2][key1] = value

    return inverse_dict


def matrix(rows, columns, alignment):
    # Prints a matrix of two sentence alignments

    '''
          wouldnt  you   know    it
    kommen|  x   |      |      |      |
    kommen|  x   |      |      |      |
    kommen|  x   |      |      |      |
    es    |      |      |      |  x   |
    '''

    longest_row = max(len(x) for x in rows)
    longest_column = max(len(x) for x in columns)

    # Header
    matrix = " " * longest_column
    for row in rows:
        matrix += "{word:^{width}}".format(word=row, width=longest_row)
    matrix += "\n"

    # Body
    for (i, column) in enumerate(columns, 1):
        matrix += "{word:<{width}}".format(word=column, width=longest_column)
        matrix += "|"
        for (j, row) in enumerate(rows, 1):
            if (j, i) in alignment:
                marker = "x"
            else:
                marker = " "
            matrix += "{word:^{width}}".format(word=marker, width=longest_row - 1)
            matrix += "|"
        matrix += "\n"
    return matrix


class HashableDict(dict):
    """
    This class implements a hashable dict, which can be
    put into a set.
    """

    def __key(self):
        return tuple((k, self[k]) for k in self)

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()
