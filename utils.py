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
            for rv in iterate_nested_dict(value, keys + (key, )):
                yield rv
    else:
        yield (keys, dict)


def matrix(rows_text, columns_text, alignment):

    """
    m: row
    n: column
    lst: items
    |x| | |
    | | |x|
    """
    rows = tokenize(rows_text)
    columns = tokenize(columns_text)

    longest_row = max(len(x) for x in rows)
    longest_column = max(len(x) for x in columns)

    # Header
    matrix = " " * longest_column
    for row in rows:
        matrix += "{word:^{width}}".format(word=row, width=longest_row)
    matrix += "\n"

    # Body
    for (i, row) in enumerate(rows):
        matrix += "{word:<{width}}".format(word=columns[i], width=longest_column)
        matrix += "|"
        for (j, column) in enumerate(columns):
            if (j, i) in alignment:
                marker = "x"
            else:
                marker = " "
            matrix += "{word:^{width}}".format(word=marker, width=longest_row - 1)
            matrix += "|"
        matrix += "\n"
    return matrix