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
