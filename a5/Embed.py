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
        self.filters = nn.ParameterList([nn.Parameter(torch.rand(1, 1, vocab_size-dim+i)) for i in range(1, dim+1)])
        self.w_t = nn.Parameter(torch.rand(dim, dim))
        self.w_h = nn.Parameter(torch.rand(dim, dim))
        self.dim = dim
        self.vocab_size = vocab_size
    def forward(self, input):
        one_hot = torch.zeros(self.vocab_size)
        one_hot[0] = torch.tensor(1)
        one_hot = one_hot.view(1, 1, -1)
        y_k = torch.zeros(self.dim)
        for i in range(y_k.size()[0]):
            y_k[i] = torch.max(F.conv1d(one_hot, self.filters[i]))
        t = F.sigmoid(F.linear(y_k, self.w_t))
        g = F.relu(F.linear(y_k, self.w_h))
        z = t*g + (1-t)*y_k

        return F.log_softmax(z, dim=0)
