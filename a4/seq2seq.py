#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This code is based on the tutorial by Sean Robertson <https://github.com/spro/practical-pytorch> found here:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

Students *MAY NOT* view the above tutorial or use it as a reference in any way.
"""


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


logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s')

# we are forcing the use of cpu, if you have access to a gpu, you can set the flag to "cuda"
# make sure you are very careful if you are using a gpu on a shared cluster/grid,
# it can be very easy to confict with other people's jobs.
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

SOS_token = "<SOS>"
EOS_token = "<EOS>"

SOS_index = 0
EOS_index = 1
MAX_LENGTH = 15


class Vocab:
    """ This class handles the mapping between the words and their indicies
    """
    def __init__(self, lang_code):
        self.lang_code = lang_code
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_index: SOS_token, EOS_index: EOS_token}
        self.n_words = 2  # Count SOS and EOS

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self._add_word(word)

    def _add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


######################################################################


def split_lines(input_file):
    """split a file like:
    first src sentence|||first tgt sentence
    second src sentence|||second tgt sentence
    into a list of things like
    [("first src sentence", "first tgt sentence"),
     ("second src sentence", "second tgt sentence")]
    """
    logging.info("Reading lines of %s...", input_file)
    # Read the file and split into lines
    lines = open(input_file, encoding='utf-8').read().strip().split('\n')
    # Split every line into pairs
    pairs = [l.split('|||') for l in lines]
    return pairs


def make_vocabs(src_lang_code, tgt_lang_code, train_file):
    """ Creates the vocabs for each of the langues based on the training corpus.
    """
    src_vocab = Vocab(src_lang_code)
    tgt_vocab = Vocab(tgt_lang_code)

    train_pairs = split_lines(train_file)

    for pair in train_pairs:
        src_vocab.add_sentence(pair[0])
        tgt_vocab.add_sentence(pair[1])

    logging.info('%s (src) vocab size: %s', src_vocab.lang_code, src_vocab.n_words)
    logging.info('%s (tgt) vocab size: %s', tgt_vocab.lang_code, tgt_vocab.n_words)

    return src_vocab, tgt_vocab

######################################################################

def tensor_from_sentence(vocab, sentence):
    """creates a tensor from a raw sentence
    """
    indexes = []
    for word in sentence.split():
        try:
            indexes.append(vocab.word2index[word])
        except KeyError:
            pass
            # logging.warn('skipping unknown subword %s. Joint BPE can produces subwords at test time which are not in vocab. As long as this doesnt happen every sentence, this is fine.', word)
    indexes.append(EOS_index)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensors_from_pair(src_vocab, tgt_vocab, pair):
    """creates a tensor from a raw sentence pair
    """
    input_tensor = tensor_from_sentence(src_vocab, pair[0])
    target_tensor = tensor_from_sentence(tgt_vocab, pair[1])
    return input_tensor, target_tensor


######################################################################

class LSTM:
    def __init__(self, m, n, n_prime=0, encode=True):
        if encode:
            #W refers to an input->hidden matrix
            #U refers to a hidden->hidden matrix
            #b refers to bias
            #all randomly initialized
            #let c_o and h_o be initialized as 0
            self.input_size = m
            self.hidden_size = n

            self.W_f = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_i = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_o = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_c = torch.rand(m, n, dtype=torch.double, requires_grad=True)

            self.U_f = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_i = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_o = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_c = torch.rand(n, n, dtype=torch.double, requires_grad=True)

            self.b_f = torch.rand(n, 1, dtype=torch.double, requires_grad=True)
            self.b_i = torch.rand(n, 1, dtype=torch.double, requires_grad=True)
            self.b_o = torch.rand(n, 1, dtype=torch.double, requires_grad=True)
            self.b_c = torch.rand(n, 1, dtype=torch.double, requires_grad=True)

            self.c = torch.tensor(0, dtype=torch.double)
            self.h = torch.tensor(0, dtype=torch.double)

            #for convenience, idk if we need it
            self.matrix_map = {"Wf": self.W_f, "Wi": self.W_i, "Wo": self.W_f, "Wc": self.W_c, "Uf": self.U_f,
                          "Ui": self.U_i, "Uo": self.U_o, "Uc": self.U_c, "bf": self.b_f, "bi": self.b_i,
                          "bo": self.b_o, "bc": self.b_c, "c": self.c, "h": self.h}
        else:
            l = n*2 #this is weird, just roll with it
            self.output_size = m
            self.hidden_size = n
            self.output_length = n_prime #num tokens in output

            self.s = None #initialized as None here (really it will be W_s*h_1)
            self.alpha = None #attention weights that need to be accessed

            self.W = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_z = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_r = torch.rand(m, n, dtype=torch.double, requires_grad=True)
            self.W_o = torch.rand(m, l, dtype=torch.double, requires_grad=True)

            self.U = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_z = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_r = torch.rand(n, n, dtype=torch.double, requires_grad=True)
            self.U_o = torch.rand(2*l, n, dtype=torch.double, requires_grad=True)

            self.C = torch.rand(2*n, n, dtype=torch.double, requires_grad=True)
            self.C_z = torch.rand(2*n, n, dtype=torch.double, requires_grad=True)
            self.C_r = torch.rand(2*n, n, dtype=torch.double, requires_grad=True)
            self.C_o = torch.rand(2*l, 2*n, dtype=torch.double, requires_grad=True)

            self.V_o = torch.rand(2*l, m, dtype=torch.double, requires_grad=True)

            self.W_s = torch.rand(n, n, dtype=torch.double, requires_grad=True) #for initializing hidden state with h1

            #for attention calculation
            self.V_a = torch.rand(n_prime, 1, dtype=torch.double, requires_grad=True)
            self.W_a = torch.rand(n_prime, n, dtype=torch.double, requires_grad=True)
            self.U_a = torch.rand(n_prime, 2*n, dtype=torch.double, requires_grad=True)

            #for convenience, idk if we need it
            self.matrix_map = {"s": self.s, "alpha": self.alpha, "W": self.W, "Wz": self.W_z, "Wr": self.W_r,
                               "Wo": self.W_o, "U": self.U, "Uz": self.U_z, "Ur": self.U_r, "Uo": self.U_o,  "C": self.C,
                               "Cz": self.C_z, "Cr": self.C_r, "Co": self.C_o, "Vo": self.V_o, "Ws": self.W_s, "Va": self.V_a,
                               "Wa": self.W_a, "Ua": self.U_a}

    def forwardEncode(self, input, hidden, weights):
        #preprocessing
        input = torch.t(input)
        #hidden = hidden.view(-1, hidden.shape[0])
        hidden = torch.t(hidden)

        f_t = torch.sigmoid(torch.matmul(torch.t(weights["Wf"]), input)
                            + torch.matmul(weights["Uf"], hidden)
                            + weights["bf"])
        i_t = torch.sigmoid(torch.matmul(torch.t(weights["Wi"]), input)
                            + torch.matmul(weights["Ui"], hidden)
                            + weights["bi"])
        o_t = torch.sigmoid(torch.matmul(torch.t(weights["Wo"]), input)
                            + torch.matmul(weights["Uo"], hidden)
                            + weights["bo"])
        #here, we're computing an update to c_t with the previous value of c_t, initialized at 0
        weights["c"] = f_t * weights["c"] + i_t * torch.tanh(torch.matmul(torch.t(weights["Wc"]), input)
                                                    + torch.matmul(weights["Uc"], hidden)
                                                    + weights["bc"])
        return o_t * torch.tanh(weights["c"]) #double check the activation function

    def get_c(self, s, h, weights):
        #s is a singel hidden state
        #h is an array of all encoder-contexts (also confusingly called hidden states)
        e = torch.zeros(h.shape[0], dtype=torch.double)
        for i in range(h.shape[0]):
            temp = torch.matmul(weights["Wa"], s) + torch.matmul(weights["Ua"], h[i]).view(weights["Ua"].shape[0] ,-1)
            e[i] = torch.exp(torch.matmul(torch.t(weights["Va"]), torch.tanh(temp)))

        #normalizing (softmax)
        sum_norm = torch.sum(e)
        for i in range(e.shape[0]):
            e[i] /= sum_norm #THESE ARE NOW OUR APHAS
        for i in range(e.shape[0]):
            h[i] *= e[i] #we are computing the array that will sum to our c_i

        weights["alpha"] = h #saving attention weights
        return torch.sum(h, dim=0) #this is c_i from the paper (the context!)

    def get_r(self, y, s, c, weights):
        return torch.sigmoid(torch.matmul(torch.t(weights["Wr"]), y)
                             + torch.matmul(torch.t(weights["Ur"]), s)
                             + torch.matmul(torch.t(weights["Cr"]), c))

    def get_z(self, y, s, c, weights):
        return torch.sigmoid(torch.matmul(torch.t(weights["Wz"]), y)
                             + torch.matmul(torch.t(weights["Uz"]), s)
                             + torch.matmul(torch.t(weights["Cz"]), c))

    def get_s_tidle(self, y, s, c, weights):
        return torch.sigmoid(torch.matmul(torch.t(weights["W"]), y)
                             + torch.matmul(torch.t(weights["U"]), self.get_r(y, s, c, weights)*s)
                             + torch.matmul(torch.t(weights["C"]), c))
    def get_s(self, y, s, c, weights):
        z = self.get_z(y, s, c, weights)
        s_tilde = self.get_s_tidle(y, s, c, weights)
        weights["s"] = (1 - z) * s + z * s_tilde
        return weights["s"]

    def get_t_tilde(self, y, s, c, weights):
        s = self.get_s(y, s, c, weights)
        return torch.sigmoid(torch.matmul(weights["Uo"], s)
                             + torch.matmul(weights["Vo"], y)
                             + torch.matmul(weights["Co"], c))
    def get_t(self, y, s, c, weights):
        t_tilde = self.get_t_tilde(y, s, c, weights)
        t = torch.zeros((int(t_tilde.shape[0]/2), 1), dtype=torch.double)
        for i in range(t.shape[0]):
            t[i] = torch.max(t_tilde[2*i], t_tilde[2*i+1])
        return t

    def forwardDecode(self, y, s, c, weights):
        #y is output
        #s is hidden
        #c is context
        c = c.view(c.shape[0], -1)
        t = self.get_t(y, s, c, weights)
        return torch.matmul(weights["Wo"], t)

class EncoderRNN(nn.Module):
    """the class for the enoder RNN
    """
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        """Initilize a word embedding and bi-directional LSTM encoder
        For this assignment, you should *NOT* use nn.LSTM.
        Instead, you should implement the equations yourself.
        See, for example, https://en.wikipedia.org/wiki/Long_short-term_memory#LSTM_with_a_forget_gate
        You should make your LSTM modular and re-use it in the Decoder.
        """
        "*** YOUR CODE HERE ***"
        #We are choosing to initialize the embeddings during training itself
        # This is so that we only worry about the embedded vectors as inputs and outputs for the sake of the
        # encoder-decoder. For example, an input sentence's words would be embedded, and then the list of tensors
        # would be fed into the encoder. Upon output from the decoder, the embeddings would be decoded.

        #bi-directional
        self.lstm_L2R = LSTM(input_size, hidden_size)
        self.lstm_R2L = LSTM(input_size, hidden_size)

        self.W_f_1 = nn.Parameter(self.lstm_L2R.matrix_map["Wf"], requires_grad=True)
        self.W_i_1 = nn.Parameter(self.lstm_L2R.matrix_map["Wi"], requires_grad=True)
        self.W_o_1 = nn.Parameter(self.lstm_L2R.matrix_map["Wo"], requires_grad=True)
        self.W_c_1 = nn.Parameter(self.lstm_L2R.matrix_map["Wc"], requires_grad=True)
        self.U_f_1 = nn.Parameter(self.lstm_L2R.matrix_map["Uf"], requires_grad=True)
        self.U_i_1 = nn.Parameter(self.lstm_L2R.matrix_map["Ui"], requires_grad=True)
        self.U_o_1 = nn.Parameter(self.lstm_L2R.matrix_map["Uo"], requires_grad=True)
        self.U_c_1 = nn.Parameter(self.lstm_L2R.matrix_map["Uc"], requires_grad=True)
        self.b_f_1 = nn.Parameter(self.lstm_L2R.matrix_map["bf"], requires_grad=True)
        self.b_i_1 = nn.Parameter(self.lstm_L2R.matrix_map["bi"], requires_grad=True)
        self.b_o_1 = nn.Parameter(self.lstm_L2R.matrix_map["bo"], requires_grad=True)
        self.b_c_1 = nn.Parameter(self.lstm_L2R.matrix_map["bc"], requires_grad=True)
        self.c_1 = self.lstm_L2R.matrix_map["c"]
        self.h_1 = self.lstm_L2R.matrix_map["h"]

        self.weights_1 = {"Wf": self.W_f_1, "Wi": self.W_i_1, "Wo": self.W_f_1, "Wc": self.W_c_1, "Uf": self.U_f_1,
                      "Ui": self.U_i_1, "Uo": self.U_o_1, "Uc": self.U_c_1, "bf": self.b_f_1, "bi": self.b_i_1,
                      "bo": self.b_o_1, "bc": self.b_c_1, "c": self.c_1, "h": self.h_1}

        self.W_f_2 = nn.Parameter(self.lstm_R2L.matrix_map["Wf"], requires_grad=True)
        self.W_i_2 = nn.Parameter(self.lstm_R2L.matrix_map["Wi"], requires_grad=True)
        self.W_o_2 = nn.Parameter(self.lstm_R2L.matrix_map["Wo"], requires_grad=True)
        self.W_c_2 = nn.Parameter(self.lstm_R2L.matrix_map["Wc"], requires_grad=True)
        self.U_f_2 = nn.Parameter(self.lstm_R2L.matrix_map["Uf"], requires_grad=True)
        self.U_i_2 = nn.Parameter(self.lstm_R2L.matrix_map["Ui"], requires_grad=True)
        self.U_o_2 = nn.Parameter(self.lstm_R2L.matrix_map["Uo"], requires_grad=True)
        self.U_c_2 = nn.Parameter(self.lstm_R2L.matrix_map["Uc"], requires_grad=True)
        self.b_f_2 = nn.Parameter(self.lstm_L2R.matrix_map["bf"], requires_grad=True)
        self.b_i_2 = nn.Parameter(self.lstm_L2R.matrix_map["bi"], requires_grad=True)
        self.b_o_2 = nn.Parameter(self.lstm_L2R.matrix_map["bo"], requires_grad=True)
        self.b_c_2 = nn.Parameter(self.lstm_L2R.matrix_map["bc"], requires_grad=True)
        self.c_2 = self.lstm_R2L.matrix_map["c"]
        self.h_2 = self.lstm_R2L.matrix_map["h"]

        self.weights_2 = {"Wf": self.W_f_2, "Wi": self.W_i_2, "Wo": self.W_f_2, "Wc": self.W_c_2, "Uf": self.U_f_2,
                      "Ui": self.U_i_2, "Uo": self.U_o_2, "Uc": self.U_c_2, "bf": self.b_f_2, "bi": self.b_i_2,
                      "bo": self.b_o_2, "bc": self.b_c_2, "c": self.c_2, "h": self.h_2}




    def forward(self, input, hidden):
        """runs the forward pass of the encoder
        returns the output and the hidden state
        """
        "*** YOUR CODE HERE ***"
        #makes deep copies of tensor (dw, I checked)
        #print(hidden.shape)
        hiddenL2R = hidden.view(-1, hidden.shape[0], hidden.shape[1])
        hiddenR2L = hidden.view(-1, hidden.shape[0], hidden.shape[1])
        for i in range(len(input)):
            hiddenL2R = torch.cat((hiddenL2R, self.lstm_L2R.forwardEncode(input[i], hiddenL2R[-1], self.weights_1).view(-1, hidden.shape[0], hidden.shape[1])))
            hiddenR2L = torch.cat((hiddenR2L, self.lstm_L2R.forwardEncode(input[len(input) - i - 1], hiddenR2L[-1], self.weights_2).view(-1, hidden.shape[0], hidden.shape[1])))
        combined = torch.zeros((hiddenL2R.shape[0],hiddenL2R.shape[2]*2), dtype=torch.double)
        for i in range(combined.shape[0]):
            combined[i] = torch.cat((hiddenL2R[i][0], hiddenR2L[i][0]))
        #raise NotImplementedError
        return combined

    def rachet_unit_test(self):
        input = torch.tensor([[i for i in range(self.input_size)]], dtype=torch.double)
        hidden = torch.tensor([[i*2 for i in range(self.hidden_size)]], dtype=torch.double)
        return self.forward([input, input, input], hidden)

    def get_initial_hidden_state(self):
        return torch.zeros(self.hidden_size, 1, device=device)


class AttnDecoderRNN(nn.Module):
    """the class for the decoder
    """
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.dropout = nn.Dropout(self.dropout_p)

        """Initilize your word embedding, decoder LSTM, and weights needed for your attention here
        """
        "*** YOUR CODE HERE ***"
        #self.embedding = torch.zeros(1, output_size, dtype=torch.double)
        #we don't initialize an embedding here since we're not taking in raw input
        self.decoder = LSTM(output_size, hidden_size, n_prime=max_length, encode=False)

        self.s = nn.Parameter(self.decoder.matrix_map["s"], requires_grad=True)
        self.alpha = nn.Parameter(self.decoder.matrix_map["alpha"], requires_grad=True)

        self.W = nn.Parameter(self.decoder.matrix_map["W"], requires_grad=True)
        self.W_z = nn.Parameter(self.decoder.matrix_map["Wz"], requires_grad=True)
        self.W_r = nn.Parameter(self.decoder.matrix_map["Wr"], requires_grad=True)
        self.W_o = nn.Parameter(self.decoder.matrix_map["Wo"], requires_grad=True)

        self.U = nn.Parameter(self.decoder.matrix_map["U"], requires_grad=True)
        self.U_z = nn.Parameter(self.decoder.matrix_map["Uz"], requires_grad=True)
        self.U_r = nn.Parameter(self.decoder.matrix_map["Ur"], requires_grad=True)
        self.U_o = nn.Parameter(self.decoder.matrix_map["Uo"], requires_grad=True)

        self.C = nn.Parameter(self.decoder.matrix_map["C"], requires_grad=True)
        self.C_z = nn.Parameter(self.decoder.matrix_map["Cz"], requires_grad=True)
        self.C_r = nn.Parameter(self.decoder.matrix_map["Cr"], requires_grad=True)
        self.C_o = nn.Parameter(self.decoder.matrix_map["Co"], requires_grad=True)

        self.V_o = nn.Parameter(self.decoder.matrix_map["Vo"], requires_grad=True)

        self.W_s = nn.Parameter(self.decoder.matrix_map["Ws"], requires_grad=True)

        self.V_a = nn.Parameter(self.decoder.matrix_map["Va"], requires_grad=True)
        self.W_a = nn.Parameter(self.decoder.matrix_map["Wa"], requires_grad=True)
        self.U_a = nn.Parameter(self.decoder.matrix_map["Ua"], requires_grad=True)

        self.weights = {"s": self.s, "alpha": self.alpha, "W": self.W, "Wz": self.W_z, "Wr": self.W_r,
                           "Wo": self.W_o, "U": self.U, "Uz": self.U_z, "Ur": self.U_r, "Uo": self.U_o,  "C": self.C,
                           "Cz": self.C_z, "Cr": self.C_r, "Co": self.C_o, "Vo": self.V_o, "Ws": self.W_s, "Va": self.V_a,
                           "Wa": self.W_a, "Ua": self.U_a}


    def forward(self, output, hidden, encoder_outputs):
        """runs the forward pass of the decoder
        returns the log_softmax, hidden state, and attn_weights

        Dropout (self.dropout) should be applied to the word embeddings.
        """

        "*** YOUR CODE HERE ***"
        context = self.decoder.get_c(hidden, encoder_outputs, self.weights) #calculate context
        log_softmax = torch.log(torch.softmax(self.decoder.forwardDecode(output, hidden, context, self.weights), dim=0)) #do forward pass
        self.dropout(log_softmax)

        #now we can get the next hidden state and attention weight since we've done a forward pass
        hidden = self.decoder.s
        attn_weights = self.decoder.alpha
        return log_softmax, hidden, attn_weights

    def get_initial_hidden_state(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


######################################################################

def train(input_tensor, target_tensor, encoder, decoder, optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.get_initial_hidden_state()

    # make sure the encoder and decoder are in training mode so dropout is applied
    encoder.train()
    decoder.train()

    "*** YOUR CODE HERE ***"
    embed = nn.Embedding(encoder.input_size, encoder.input_size)
    print(embed(input_tensor))
    print(target_tensor)

    return loss.item()



######################################################################

def translate(encoder, decoder, sentence, src_vocab, tgt_vocab, max_length=MAX_LENGTH):
    """
    runs tranlsation, returns the output and attention
    """

    # switch the encoder and decoder to eval mode so they are not applying dropout
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        input_tensor = tensor_from_sentence(src_vocab, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.get_initial_hidden_state()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_index]], device=device)

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_index:
                decoded_words.append(EOS_token)
                break
            else:
                decoded_words.append(tgt_vocab.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


######################################################################

# Translate (dev/test)set takes in a list of sentences and writes out their transaltes
def translate_sentences(encoder, decoder, pairs, src_vocab, tgt_vocab, max_num_sentences=None, max_length=MAX_LENGTH):
    output_sentences = []
    for pair in pairs[:max_num_sentences]:
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        output_sentences.append(output_sentence)
    return output_sentences


######################################################################
# We can translate random sentences  and print out the
# input, target, and output to make some subjective quality judgements:
#

def translate_random_sentence(encoder, decoder, pairs, src_vocab, tgt_vocab, n=1):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = translate(encoder, decoder, pair[0], src_vocab, tgt_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


######################################################################

def show_attention(input_sentence, output_words, attentions):
    """visualize the attention mechanism. And save it to a file.
    Plots should look roughly like this: https://i.stack.imgur.com/PhtQi.png
    You plots should include axis labels and a legend.
    you may want to use matplotlib.
    """

    "*** YOUR CODE HERE ***"
    raise NotImplementedError


def translate_and_show_attention(input_sentence, encoder1, decoder1, src_vocab, tgt_vocab):
    output_words, attentions = translate(
        encoder1, decoder1, input_sentence, src_vocab, tgt_vocab)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


def clean(strx):
    """
    input: string with bpe, EOS
    output: list without bpe, EOS
    """
    return ' '.join(strx.replace('@@ ', '').replace(EOS_token, '').strip().split())


######################################################################

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--hidden_size', default=256, type=int,
                    help='hidden size of encoder/decoder, also word vector size')
    ap.add_argument('--n_iters', default=100000, type=int,
                    help='total number of examples to train on')
    ap.add_argument('--print_every', default=5000, type=int,
                    help='print loss info every this many training examples')
    ap.add_argument('--checkpoint_every', default=10000, type=int,
                    help='write out checkpoint every this many training examples')
    ap.add_argument('--initial_learning_rate', default=0.001, type=int,
                    help='initial learning rate')
    ap.add_argument('--src_lang', default='fr',
                    help='Source (input) language code, e.g. "fr"')
    ap.add_argument('--tgt_lang', default='en',
                    help='Source (input) language code, e.g. "en"')
    ap.add_argument('--train_file', default='data/fren.train.bpe',
                    help='training file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--dev_file', default='data/fren.dev.bpe',
                    help='dev file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence')
    ap.add_argument('--test_file', default='data/fren.test.bpe',
                    help='test file. each line should have a source sentence,' +
                         'followed by "|||", followed by a target sentence' +
                         ' (for test, target is ignored)')
    ap.add_argument('--out_file', default='out.txt',
                    help='output file for test translations')
    ap.add_argument('--load_checkpoint', nargs=1,
                    help='checkpoint file to start from')

    args = ap.parse_args()

    # process the training, dev, test files

    # Create vocab from training data, or load if checkpointed
    # also set iteration
    if args.load_checkpoint is not None:
        state = torch.load(args.load_checkpoint[0])
        iter_num = state['iter_num']
        src_vocab = state['src_vocab']
        tgt_vocab = state['tgt_vocab']
    else:
        iter_num = 0
        src_vocab, tgt_vocab = make_vocabs(args.src_lang,
                                           args.tgt_lang,
                                           args.train_file)

    encoder = EncoderRNN(src_vocab.n_words, args.hidden_size).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, tgt_vocab.n_words, dropout_p=0.1).to(device)

    # encoder/decoder weights are randomly initilized
    # if checkpointed, load saved weights
    if args.load_checkpoint is not None:
        encoder.load_state_dict(state['enc_state'])
        decoder.load_state_dict(state['dec_state'])

    # read in datafiles
    train_pairs = split_lines(args.train_file)
    dev_pairs = split_lines(args.dev_file)
    test_pairs = split_lines(args.test_file)

    # set up optimization/loss
    params = list(encoder.parameters()) + list(decoder.parameters())  # .parameters() returns generator
    optimizer = optim.Adam(params, lr=args.initial_learning_rate)
    criterion = nn.NLLLoss()

    # optimizer may have state
    # if checkpointed, load saved state
    if args.load_checkpoint is not None:
        optimizer.load_state_dict(state['opt_state'])

    start = time.time()
    print_loss_total = 0  # Reset every args.print_every

    while iter_num < args.n_iters:
        iter_num += 1
        training_pair = tensors_from_pair(src_vocab, tgt_vocab, random.choice(train_pairs))
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder,
                     decoder, optimizer, criterion)
        print_loss_total += loss

        if iter_num % args.checkpoint_every == 0:
            state = {'iter_num': iter_num,
                     'enc_state': encoder.state_dict(),
                     'dec_state': decoder.state_dict(),
                     'opt_state': optimizer.state_dict(),
                     'src_vocab': src_vocab,
                     'tgt_vocab': tgt_vocab,
                     }
            filename = 'state_%010d.pt' % iter_num
            torch.save(state, filename)
            logging.debug('wrote checkpoint to %s', filename)

        if iter_num % args.print_every == 0:
            print_loss_avg = print_loss_total / args.print_every
            print_loss_total = 0
            logging.info('time since start:%s (iter:%d iter/n_iters:%d%%) loss_avg:%.4f',
                         time.time() - start,
                         iter_num,
                         iter_num / args.n_iters * 100,
                         print_loss_avg)
            # translate from the dev set
            translate_random_sentence(encoder, decoder, dev_pairs, src_vocab, tgt_vocab, n=2)
            translated_sentences = translate_sentences(encoder, decoder, dev_pairs, src_vocab, tgt_vocab)

            references = [[clean(pair[1]).split(), ] for pair in dev_pairs[:len(translated_sentences)]]
            candidates = [clean(sent).split() for sent in translated_sentences]
            dev_bleu = corpus_bleu(references, candidates)
            logging.info('Dev BLEU score: %.2f', dev_bleu)

    # translate test set and write to file
    translated_sentences = translate_sentences(encoder, decoder, test_pairs, src_vocab, tgt_vocab)
    with open(args.out_file, 'wt', encoding='utf-8') as outf:
        for sent in translated_sentences:
            outf.write(clean(sent) + '\n')

    # Visualizing Attention
    translate_and_show_attention("on p@@ eu@@ t me faire confiance .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("j en suis contente .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("vous etes tres genti@@ ls .", encoder, decoder, src_vocab, tgt_vocab)
    translate_and_show_attention("c est mon hero@@ s ", encoder, decoder, src_vocab, tgt_vocab)


if __name__ == '__main__':
    main()
