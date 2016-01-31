# -*- coding: utf-8 -*-
import csv
import regex

class NewsCorpus():

    REMOVE_PUNCTUATION = regex.compile("[^\w\s]")

    def __init__(self):

        self.sentence_pairs = []
        reader = csv.reader(file("news_1000.csv", 'rU'))

        for row in reader:

            # some rows are incorrectly exported/parsed
            if len(row) != 2:
                continue

            if self.filter_illegal(row):
                decoded = [string.decode('utf-8') for string in row]
                self.sentence_pairs.append(tuple(decoded))

    def get_sentence_pairs(self):
        return self.sentence_pairs

    def get_test_sentence_pairs(self):
        return self.sentence_pairs[:10]

    def filter_illegal(self, row):

        for string in row:
            if len(NewsCorpus.REMOVE_PUNCTUATION.sub("", string.strip())) < 5:
                return False

        return True
