#!/usr/bin/env python
import optparse
import sys
import models
import math
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


def notAlreadyTranslated(a, j, k):
    if a is None:
        return True

    for index in a:
        for index2 in range(j, k):
            if index == index2:
                return False
    return True


for f in french:
    # The following code implements a monotone decoding
    # algorithm (one that doesn't permute the target phrases).
    # Hence all hypotheses in stacks[i] represent translations of
    # the first i words of the input sentence. You should generalize
    # this so that they can represent translations of *any* i words.
    hypothesis = namedtuple(
        "hypothesis", "logprob, lm_state, predecessor, phrase, wordsTranslated")
    initial_hypothesis = hypothesis(0.0, lm.begin(), None, None, None)
    stacks = [{} for _ in f] + [{}]
    stacks[0][lm.begin()] = initial_hypothesis
    threshold = 0.005
    for i, stack in enumerate(stacks[:-1]):
        # This is what we had for Threshold Pruning but it skewed our results so we decided to omit it
        # bestLogProb = -100000000
        # for h in sorted(stack.itervalues()):
            # bestLogProb = h.logprob if h.logprob > bestLogProb else bestLogProb
        for h in sorted(stack.itervalues(), key=lambda h: -h.logprob)[:opts.s]:
            # if h.logprob < bestLogProb * 1 / threshold: #threshold pruning on the stack of hypotheses
            #     continue
            for j in xrange(0, len(f)):
                for k in xrange(j + 1, len(f) + 1):
                    if f[j:k] in tm and notAlreadyTranslated(h.wordsTranslated, j, k):
                        for phrase in tm[f[j:k]]:
                            distance = abs(k - i - 1)
                            logprob = h.logprob + phrase.logprob
                            # Similarily when we chose to penalize hypothesis with bigger distance swapping we got poor results
                            # logprob += 0 if distance == 0 else math.log(distance)
                            lm_state = h.lm_state
                            for word in phrase.english.split():
                                (lm_state, word_logprob) = lm.score(
                                    lm_state, word)
                                logprob += word_logprob
                            logprob += lm.end(lm_state) if k == len(f) else 0.0
                            new_words_translated = range(
                                j, k) if h.wordsTranslated is None else h.wordsTranslated + range(j, k)
                            new_hypothesis = hypothesis(
                                logprob, lm_state, h, phrase, new_words_translated)
                            # second case is recombination
                            if lm_state not in stacks[len(new_hypothesis.wordsTranslated)] or stacks[len(new_hypothesis.wordsTranslated)][lm_state].logprob < logprob:
                                stacks[len(new_hypothesis.wordsTranslated)
                                       ][lm_state] = new_hypothesis

    winner = max(stacks[-1].itervalues(), key=lambda h: h.logprob)

    def extract_english(h):
        return "" if h.predecessor is None else "%s%s " % (extract_english(h.predecessor), h.phrase.english)
    print extract_english(winner)

    if opts.verbose:
        def extract_tm_logprob(h):
            return 0.0 if h.predecessor is None else h.phrase.logprob + extract_tm_logprob(h.predecessor)
        tm_logprob = extract_tm_logprob(winner)
        sys.stderr.write("LM = %f, TM = %f, Total = %f\n" %
                         (winner.logprob - tm_logprob, tm_logprob, winner.logprob))
