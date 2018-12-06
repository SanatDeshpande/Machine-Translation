from collections import defaultdict
import math

class Model:
    def __init__(self, filename):
        with open(filename, "r") as f:
            self.sentences = f.read().split("\n")
    def build(self):
        self.word_count = defaultdict(int)
        self.bigram_count = defaultdict(int)
        for sentence in self.sentences:
            last_word = "<s>"
            for word in sentence.split(" "):
                self.word_count[word] += 1
                self.bigram_count[(last_word, word)] += 1
                last_word = word
            self.bigram_count[(last_word, "<e>")] += 1

    def score(self, phrase):
        phrase = phrase + " <e>"
        score = 0
        last_word = "<s>"
        for word in phrase.split(" "):
            word_count = self.word_count[word]
            joint_count = self.bigram_count[(last_word, word)]
            if word_count != 0:
                score += math.log(joint_count / word_count)
            last_word = word
            print(score)
        return score




if __name__ == '__main__':
    m = Model("./data/small_english.txt")
    m.build()
    m.score("hello")
