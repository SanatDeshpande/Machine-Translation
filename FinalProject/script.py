formal_words = set(open('./data/dictionary.txt').read().split())
contractions = set(open('./data/contractions.txt').read().split())
new_f_words = formal_words.difference(contractions)

with open('./data/dictionary2.txt', 'w') as f:
    for item in new_f_words:
        f.write("%s\n" % item)
