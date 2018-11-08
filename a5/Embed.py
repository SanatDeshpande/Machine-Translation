from __future__ import unicode_literals, print_function, division

import argparse
import logging
import random
import time
from io import open

import matplotlib
#if you are running on the gradx/ugradx/ another cluster,
#you will need the following line
#if you run on a local machine, you can comment it out
matplotlib.use('agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import torch.nn.functional as F
from nltk.translate.bleu_score import corpus_bleu
from torch import optim

class Embed(nn.Module):
    def __init__(self, vocab_size, dim):
        super(Embed, self).__init__()
        self.convs = [nn.Conv1d(1, 1, kernel_size=i) for i in range(1, vocab_size+1)]
        #self.conv = nn.Conv1d(1, 1, kernel_size=1)
    def forward(self, input):
        input = input
        a = [F.max_pool1d(F.relu(convolution(input)), kernel_size=1) for convolution in self.convs]
        for i in a:
            print(i.size())
        return a

e = Embed(100, 10)
sample = torch.rand((1, 1, 100))
print(e(sample).size() , sample.size())
