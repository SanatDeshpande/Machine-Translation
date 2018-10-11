#!/usr/bin/env python
import optparse
import sys
import models
from collections import namedtuple

optparser = optparse.OptionParser()
optparser.add_option("-i", "--input", dest="input", default="data/input",
                     help="File containing sentences to translate (default=data/input)")
optparser.add_option("-t", "--translation-model", dest="tm", default="data/tm",
                     help="File containing translation model (default=data/tm)")
optparser.add_option("-l", "--language-model", dest="lm", default="data/lm",
                     help="File containing ARPA-format language model (default=data/lm)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=sys.maxint,
                     type="int", help="Number of sentences to decode (default=no limit)")
optparser.add_option("-k", "--translations-per-phrase", dest="k", default=1, type="int",
                     help="Limit on number of translations to consider per phrase (default=1)")
optparser.add_option("-s", "--stack-size", dest="s", default=1,
                     type="int", help="Maximum stack size (default=1)")
optparser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                     default=False,  help="Verbose mode (default=off)")
opts = optparser.parse_args()[0]

tm = models.TM(opts.tm, opts.k)
lm = models.LM(opts.lm)
french = [tuple(line.strip().split())
          for line in open(opts.input).readlines()[:opts.num_sents]]

# tm should translate unknown words as-is with probability 1
for word in set(sum(french, ())):
    if (word,) not in tm:
        tm[(word,)] = [models.phrase(word, 0.0)]

sys.stderr.write("Decoding %s...\n" % (opts.input,))

addPhraseSwapping = False if len(sys.argv) < 2 else True

class Node:
    def __init__(self, english, logprob):
        self.english = english
        self.logprob = logprob
        self.visited = False

class Graph:
    def __init__(self, node):
        self.node = {node.english : node}

    def add(self, node):
        if node not in self.node:
            self.node[node.english] = node

    def runTSP(self):
        pass

for f in french:
    g = Graph(Node('<s>', 0.0))
    for i in xrange(len(f)):
        for j in xrange(i+1, len(f)+1):
            if f[i:j] in tm:
                english = tm[f[i:j]][0].english
                logprob = tm[f[i:j]][0].logprob
                g.add(Node(english, logprob))
    break
