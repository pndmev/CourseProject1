import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
import numpy as np
from torch import FloatTensor, LongTensor

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
    
X_train, y_train = json.load(open("train_data.txt", "r"))

word2ind = json.load(open("word2ind.txt", "r"))
tag2ind = json.load(open("tag2ind.txt", "r"))

model = LSTM_Tagger(len(word2ind), len(tag2ind))

criterion = nn.CrossEntropyLoss(ignore_index = 0)
optimizer = optim.Adam(model.parameters())
batch_size = 64

n_batches = math.ceil(len(X_train) / batch_size)

model.load_state_dict(torch.load("C:\CourseWork\Term5\POS_3.pt"))

for epoch in range(3):
    total_loss = 0
    total_correct = 0
    total = 0
    with tqdm(total = n_batches) as progress_bar:
        for i, (X_batch, y_batch) in enumerate(batches((X_train, y_train), batch_size)):
 
            X_batch, y_batch = LongTensor(X_batch), LongTensor(y_batch)
            tags = model(X_batch, len(X_batch[0]))
            
            #lOSS
            optimizer.zero_grad()
            
            form_tags = FloatTensor(len(tags), len(tag2ind), len(X_batch[0]))
            for i in range(0, len(tags)):
                for j in range(0, len(tags[i][0])):
                    for k in range(0, len(tags[i])):
                        form_tags[i][j][k] = tags[i][k][j]
            
            loss = criterion(form_tags, y_batch)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            
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
            
            total_correct += current_correct
            total += current
            
            #OUTPUT
            progress_bar.update()
            progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format
            ("Step: ", loss.item(), float(current_correct) / current))
    
    torch.save(model.state_dict(), "C:\CourseWork\Term5\POS_" + str(epoch + 3) + ".pt")
    progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format
    ("Epoch: ", total_loss / n_batches, float(total_correct) / total))