import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np
from torch import LongTensor

import math
from tqdm import tqdm

import json
import os

np.random.seed(1)

def batches(data, batch_size):
    X, y = data                                                                
    n_seq = len(X)      

    ind = np.arange(n_seq)
    np.random.shuffle(ind)
    
    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        
        batch_ind = ind[start:end]
        
        max_seq_len = max(len(X[ind]) for ind in batch_ind)
        X_batch = np.zeros((max_seq_len, len(batch_ind)))
        y_batch = np.zeros((max_seq_len, len(batch_ind)))
        
        for i_batch, i_seq in enumerate(batch_ind):
            X_batch[:len(X[i_seq]), i_batch] = X[i_seq]
            y_batch[:len(y[i_seq]), i_batch] = y[i_seq]
            
        yield X_batch, y_batch

class LSTM_Tagger(nn.Module):
    def __init__(self, word2ind_size, tag2ind_size, emb_dim = 100, lstm_hid_dim = 128):
        super(LSTM_Tagger, self).__init__()
        self.emb = nn.Embedding(word2ind_size, emb_dim)
        self.emb2hid = nn.LSTM(emb_dim, lstm_hid_dim)
        self.hid2tag = nn.Linear(lstm_hid_dim, tag2ind_size)
        
    def forward(self, seq, batch_size):
        emb = self.emb(seq)
        lstm_out, _ = self.emb2hid(emb.view(len(seq), batch_size, -1))
        tags = func.softmax(self.hid2tag(lstm_out), dim = 2)
        return tags

os.chdir("C:\CourseWork\Term5")
    
#X_val, y_val = json.load(open("val_data.txt", "r"))
X_test, y_test = json.load(open("test_data.txt", "r"))

model = torch.load("POS_3.pt")
model.eval()

batch_size = 64
#n_batches = math.ceil(len(X_val) / batch_size)
n_batches = math.ceil(len(X_test) / batch_size)

#out_file = open("val_accuracy_3.txt", "w+")
out_file = open("test_accuracy_3.txt", "w+")

with tqdm(total = n_batches) as progress_bar:
    #for i, (X_batch, y_batch) in enumerate(batches((X_val, y_val), batch_size)):
    for i, (X_batch, y_batch) in enumerate(batches((X_test, y_test), batch_size)):
 
        X_batch, y_batch = LongTensor(X_batch), LongTensor(y_batch)
        tags = model(X_batch, len(X_batch[0]))
        
        #ACCURACY
        count = 0
        ind = []
        for i in range(0, len(tags)):
            l_ind = []
            for j in range(0, len(tags[i])):
                if (y_batch[i][j] != 0):
                    index = torch.argmax(tags[i][j])
                else:
                    index = 0
                    count += 1
                l_ind.append(index)
            ind.append(l_ind)
        ind = LongTensor(ind)
        
        current_correct = torch.sum(ind == y_batch)
        current_correct -= count
        current = y_batch.shape[0] * y_batch.shape[1] - count
        
        accuracy = float(current_correct) / current
        
        #OUTPUT
        out_file.write(str(accuracy))
        
        progress_bar.update()
        progress_bar.set_description('Accuracy = ' + str(accuracy))

out_file.close()