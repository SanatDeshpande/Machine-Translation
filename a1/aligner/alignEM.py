import time
import sys
import math
from collections import defaultdict

f = open("./data/hansards.f", "r").read().split("\n")
e = open("./data/hansards.e", "r").read().split("\n")

e_set = set([])
for e_sent in e:
    for e_word in e_sent.split(" "):
        e_set.add(e_word)

e_count = len(e_set)
init_prob = 1 / float(e_count)

t = defaultdict(lambda: init_prob)

countIterations = 0
continueIter = True
oldPerplexity = 0
perplexity = 0
while continueIter and countIterations < 10:
    countIterations += 1

    if oldPerplexity != 0:
        continueIter = (((oldPerplexity - perplexity) / oldPerplexity) >= 0.03)
    oldPerplexity = perplexity

    count = defaultdict(int)
    total = defaultdict(int)
    for e_sent, f_sent in zip(e,f):
        s_total = defaultdict(int)
        for e_word in e_sent.split(" "):
            s_total[e_word] = 0
            for f_word in f_sent.split(" "):
                s_total[e_word] += t[(e_word, f_word)]
        for e_word in e_sent.split(" "):
            for f_word in f_sent.split(" "):
                tmp = t[(e_word, f_word)]
                count[(e_word, f_word)] += tmp / s_total[e_word]
                total[f_word] += tmp / s_total[e_word]

        for e_word in e_sent.split(" "):
            for f_word in f_sent.split(" "):
                t[(e_word, f_word)] = count[(e_word, f_word)] / total[f_word]

    perplexity = 0
    for e_sent, f_sent in zip(e,f):
        p_e_f = 1
        for e_word in e_sent.split(" "):
            temp = 0
            for f_word in f_sent.split(" "):
                temp += t[(e_word, f_word)]
            p_e_f *= temp
        perplexity += math.log(p_e_f, 2)
    perplexity = -perplexity;

for e_sent, f_sent in zip(e,f):
    for count_f, f_word in enumerate(f_sent.split(" ")):
        alignment_max = -1
        e_index_max = -1
        for count_e, e_word in enumerate(e_sent.split(" ")):
            if t[(e_word, f_word)] > alignment_max:
                alignment_max = t[(e_word, f_word)]
                e_index_max = count_e
        sys.stdout.write("%i-%i " % (count_f,e_index_max))
    sys.stdout.write("\n")
