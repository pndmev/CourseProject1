import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch import LongTensor, FloatTensor

import json
import os

np.random.seed(1)

def wseq2iseq(wseq, word2ind):
    return [word2ind.get(word, 0) for word in wseq]

def iseq2tseq(iseq, tag2ind):
    return [list(tag2ind.keys())[ind - 1] for ind in iseq]

class LSTM_Tagger(nn.Module):
    def __init__(self, emb, tag2ind_size, emb_dim = 100, lstm_hid_dim = 128):
        super(LSTM_Tagger, self).__init__()
        self.word_embeddings = nn.Embedding.from_pretrained(emb)
        self.emb2hidden = nn.LSTM(emb_dim, lstm_hid_dim, bidirectional = True)
        self.hidden2tag = nn.Linear(lstm_hid_dim * 2, tag2ind_size)
        
    def forward(self, seq, batch_size):
        emb = self.word_embeddings(seq)
        lstm_out, _ = self.emb2hidden(emb.view(len(seq), batch_size, -1))
        tags = func.softmax(self.hidden2tag(lstm_out), dim = 2)
        return tags
    
os.chdir("C:\CourseWork\Term5")
    
word2ind = json.load(open("word2ind.txt", "r"))
tag2ind = json.load(open("tag2ind.txt", "r"))
embeddings = json.load(open("embeddings.txt", "r"))
embeddings = FloatTensor(embeddings)

model = LSTM_Tagger(embeddings, len(tag2ind))
filename = "POS_Bi3"
model.load_state_dict(torch.load(filename + ".pt"))
model.eval()

for i in range(10):
    sentence = input()
    sentence = sentence.split()
    sentence = wseq2iseq(sentence, word2ind)
    
    tags = model(LongTensor(sentence), 1)
            
    ind = []
    for i in range(0, len(tags)):
        l_ind = []
        for j in range(0, len(tags[i])):
            index = torch.argmax(tags[i][j])
            l_ind.append(index)
        ind.append(l_ind)
    ind = LongTensor(ind)
    
    print(iseq2tseq(ind, tag2ind))