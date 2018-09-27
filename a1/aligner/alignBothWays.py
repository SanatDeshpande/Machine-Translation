import time
import sys
from collections import defaultdict

f = open("./data/hansards.f", "r").read().split("\n")[:1000]
e = open("./data/hansards.e", "r").read().split("\n")[:1000]

e_set = set([])
for e_sent in e:
    for e_word in e_sent.split(" "):
        e_set.add(e_word)

e_count = len(e_set)
init_prob = 1 / float(e_count)

num_iterations = 3

t = defaultdict(lambda: init_prob)

for converge in range(num_iterations):
    count = defaultdict(int)
    total = defaultdict(int)
    s_total = defaultdict(int)
    for e_sent, f_sent in zip(e,f):
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

f_set = set([])
for f_sent in f:
    for f_word in f_sent.split(" "):
        f_set.add(f_word)

f_count = len(f_set)
init_prob_inv = 1 / float(f_count)
t_inv = defaultdict(lambda: init_prob)

for converge in range(num_iterations):
    count = defaultdict(int)
    total = defaultdict(int)
    s_total = defaultdict(int)
    for e_sent, f_sent in zip(e,f):
        for f_word in f_sent.split(" "):
            s_total[f_word] = 0
            for e_word in e_sent.split(" "):
                s_total[f_word] += t_inv[(f_word, e_word)]
        for f_word in f_sent.split(" "):
            for e_word in e_sent.split(" "):
                tmp = t_inv[(f_word, e_word)]
                count[(f_word, e_word)] += tmp / s_total[f_word]
                total[e_word] += tmp / s_total[f_word]

        for f_word in f_sent.split(" "):
            for e_word in e_sent.split(" "):
                t_inv[(f_word, e_word)] = count[(f_word, e_word)] / total[e_word]

f2e = defaultdict(set)
e2f = defaultdict(set)

for e_sent, f_sent in zip(e,f):
    setf2e = set()
    for count_f, f_word in enumerate(f_sent.split(" ")):
        alignment_max = -1
        e_index_max = -1
        for count_e, e_word in enumerate(e_sent.split(" ")):
            if t[(e_word, f_word)] > alignment_max:
                alignment_max = t[(e_word, f_word)]
                e_index_max = count_e
        # sys.stdout.write("%i-%i " % (count_f,e_index_max))
        setf2e.add((count_f, e_index_max))
    f2e[(e_sent, f_sent)] = setf2e
    # sys.stdout.write("\n")

    sete2f = set()
    for count_e, e_word in enumerate(e_sent.split(" ")):
        alignment_max = -1
        f_index_max = -1
        for count_f, f_word in enumerate(f_sent.split(" ")):
            if t_inv[(f_word, e_word)] > alignment_max:
                alignment_max = t_inv[(f_word, e_word)]
                f_index_max = count_f
        # sys.stdout.write("%i-%i " % (count_e,f_index_max))
        # sete2f.add((count_e, f_index_max))
        sete2f.add((f_index_max, count_e))
    e2f[(f_sent, e_sent)] = sete2f
    # sys.stdout.write("\n")

def grow_heuristic(f2e, e2f, e, f):
    for e_sent, f_sent in zip(e,f):
        setf2e = f2e[(e_sent, f_sent)]
        sete2f = e2f[(f_sent, e_sent)]
        intersect = setf2e.intersection(sete2f)
        union = setf2e.union(sete2f)

        addition = set()
        # GROW-DIAG()
        for i in intersect:
            for x in range(-1, 2):
                for y in range(-1, 2):
                    x_new = i[0] + x
                    y_new = i[1] + y
                    if x_new >= 0 and x < len(e_sent) and y_new >= 0 and y_new < len(f_sent):
                        tmp = (i[0] + x, i[1] + y) #neighbor
                        if tmp in union:
                            firstTuple = set([j[0] for j in intersect])
                            secondTuple = set([j[1] for j in intersect])
                            if tmp[0] not in firstTuple or tmp[1] not in secondTuple:
                                addition.add(tmp)

        intersect = intersect.union(addition) #check if addition b4 or after final-and
        addition = set()

        for i in union: # final method
            firstTuple = set([j[0] for j in intersect])
            secondTuple = set([j[1] for j in intersect])
            if i[0] not in firstTuple and i[1] not in secondTuple: #check if and or or
                addition.add(i)

        intersect = intersect.union(addition)
        for i in intersect:
            sys.stdout.write("%i-%i " % (i[0], i[1]))
        sys.stdout.write("\n")



if __name__ == "__main__":
    grow_heuristic(f2e, e2f, e, f)
