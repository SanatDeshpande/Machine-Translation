import time
from collections import defaultdict

start = time.time()
f = open("./data/hansards.f", "r").read().split("\n")[:50]
e = open("./data/hansards.e", "r").read().split("\n")[:50]


f_set = set([])
e_set = set([])
for f_sent in f:
    for f_word in f_sent.split(" "):
        f_set.add(f_word)
for e_sent in e:
    for e_word in e_sent.split(" "):
        e_set.add(e_word)
print(time.time() - start)

f_count = len(f_set)
e_count = len(e_set)
init_prob = 1 / float(e_count)

#initialize table of e|f probabilities
t = defaultdict(lambda: init_prob)
print(t[("foo", "bar")])

print(time.time() - start)

for converge in range(100):
    count = {}
    total = {}

    for e_sent in e:
        for f_sent in f:
            for e_word in e_sent.split(" "):
                s_total = defaultdict(lambda: 0)
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
print(time.time() - start)

for key, value in t:
    if value > 1:
        print("nah")
