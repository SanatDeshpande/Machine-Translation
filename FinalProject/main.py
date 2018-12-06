#!/usr/bin/env python
import optparse
import sys
import math
from collections import namedtuple
from validator_collection import checkers
from model import Model

global interjections
global language_model

def _dictionary(pharse):
    pass

def _punctuation(phrase):
    pass

def _interjection(phrase):
    phrase = phrase.split()
    if phrase[0] in interjections:
        phrase = phrase[1:]
    if phrase[-1] in interjections:
        phrase = phrase[:-1]

    return " ".join(phrase)

def _pronunciation(phrase):
    pass

def _be(pharse):
    pass

def _re_tokenization(phrase):
    def needs_tokenization(word):
        return "." in word and len(word) > 1 and not checkers.is_url(word) and not checkers.is_email(word)

    def tokenize(word):
        i = word.find('.')
        return word[:i - 1] + " " + word[i] + " " + word[i+1:]

    phrase = phrase.split()
    for i in range(len(phrase)):
        if needs_tokenization(phrase[i]):
            phrase[i] = tokenize(phrase[i])
    return " ".join(phrase)

def _prefix(phrase):
    pass

def _quotation(phrase):
    pass

def _abbreviation(phraset):
    pass

def _time(phrase):
    def normalize_time(s):
        if len(s) == 4 or len(s) == 3:
            s = s[:len(s) - 2] + ' : ' + s[len(s) - 2:]
        return s

    def check_if_am_or_pm(s):
        return s.lower() == 'am' or s.lower() == 'pm'

    phrase = phrase.split()
    for i in range(len(phrase)):
        if phrase[i].lower() == 'at' and i + 1 != len(phrase):
            if phrase[i + 1].isdigit():
                phrase[i + 1] = normalize_time(phrase[i + 1])
                if len(phrase[i + 1]) == 1 and (i + 2 == len(phrase) or not check_if_am_or_pm(phrase[i + 2])):
                    phrase.insert(i + 2, 'o\'clock')

        if check_if_am_or_pm(phrase[i]):
            if phrase[i - 1].isdigit():
                phrase[i - 1] = normalize_time(phrase[i - 1])
    return " ".join(phrase)

def score(phrase, hypotheses):
    return math.exp(language_model.score(phrase)) + .15 * len(hypotheses)

def main():
    hypotheses_producers = {
        'dictionary': _dictionary,
        'punctuation': _punctuation,
        'interjection': _interjection,
        'pronunciation': _pronunciation,
        'be': _be,
        're_tokenization': _re_tokenization,
        'prefix': _prefix,
        'quotation': _quotation,
        'abbreviation': _abbreviation,
        'time': _time
    }

    optparser = optparse.OptionParser()
    optparser.add_option("-i", "--input", dest="input", default="data/input", help="File containing sentences to translate (default=data/input)")
    optparser.add_option("-s", "--stack-size", dest="s", default=1, type="int", help="Maximum stack size (default=1)")
    optparser.add_option("-n", "--num-producers", dest="num_producers", type="int", default=len(hypotheses_producers), help="The number of different hypothesis producers you wish to have running on your sentence")
    opts = optparser.parse_args()[0]

    hypothesis = namedtuple("hypothesis", "score, list_previous_producers, phrase")
    initial_hypothesis = hypothesis(0.0, [], None) #phrase should be the input
    stacks = [[] for _ in range(opts.num_producers)] + [[]]
    stacks[0].append(initial_hypothesis)

    for i, stack in enumerate(stacks[:-1]):
        for h in sorted(stack, key=lambda h: h.score): # prune
          for producer in hypotheses_producers:
            if producer not in h.list_previous_producers:
                new_list = h.list_previous_producers + [producer]
                new_phrase = hypotheses_producers[producer](h.phrase)
                new_score = score(new_phrase, new_list)
                new_hypothesis = hypothesis(new_score, new_list, new_phrase)
                stacks[i+1].append(new_hypothesis)
    winner = max(stacks[-1], key=lambda h: h.score)
    print(winner.phrase)

if __name__ == '__main__':
    interjections = set(open('./data/interjections.txt').read().split())

    language_model = Model("./data/small_english.txt")
    language_model.build()

    print(_re_tokenization("i job pm fuck"))
