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

import random

np.random.seed(1)

def batches(data, batch_size, ochar2ind_size):
    X, y = data                                                                
    n_seq = len(X)      

    ind = np.arange(n_seq)
    np.random.shuffle(ind)
    
    for start in range(0, n_seq, batch_size):
        end = min(start + batch_size, n_seq)
        
        batch_ind = ind[start:end]
        
        max_x_len = max(len(X[ind]) for ind in batch_ind)
        max_y_len = max(len(y[ind]) for ind in batch_ind)
        X_batch = np.empty((max_x_len, len(batch_ind)))
        X_batch.fill(ochar2ind_size - 1)
        y_batch = np.empty((max_y_len, len(batch_ind)))
        y_batch.fill(ochar2ind_size - 1)
        
        for i_batch, i_seq in enumerate(batch_ind):
            X_batch[:len(X[i_seq]), i_batch] = X[i_seq]
            y_batch[:len(y[i_seq]), i_batch] = y[i_seq]
            
        yield X_batch, y_batch

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim = 20, hid_dim = 32):
        super(Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.emb = nn.Embedding(input_dim, emb_dim)
        self.emb2hid = nn.LSTM(emb_dim, hid_dim)
        
    def forward(self, exp):
        emb = self.emb(exp)     
        out, hid = self.emb2hid(emb)
        return hid

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim = 20, hid_dim = 32):
        super(Decoder, self).__init__()
        self.emb = nn.Embedding(output_dim, emb_dim)
        self.emb2hid = nn.LSTM(emb_dim, hid_dim)
        self.hid2out = nn.Linear(hid_dim, output_dim)
        
    def forward(self, input, hidden):        
        emb = self.emb(input.view(1, -1))
        out, hid = self.emb2hid(emb, hidden)        
        res = func.softmax(self.hid2out(out.squeeze(0)), dim = 1)
        return res, hid

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, exp, trg, ochar2ind_size, teacher_forcing_ratio = 0.5):       
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = ochar2ind_size
        
        out = torch.zeros(trg_len, batch_size, trg_vocab_size)
        en_hid = self.encoder(exp)        
        input = trg[0]
        
        de_hid = en_hid
        for t in range(1, trg_len):
            de_out, de_hid = self.decoder(input, de_hid)
            out[t] = de_out
            teacher_force = random.random() < teacher_forcing_ratio
            
            top = de_out.argmax(1)
            input = trg[t] if teacher_force else top
        
        return out

os.chdir("C:\CourseWork\Term5")
    
X_train, y_train = json.load(open("EXP_train_data.txt", "r"))

ichar2ind = json.load(open("EXP_ichar2ind.txt", "r"))
ochar2ind = json.load(open("EXP_ochar2ind.txt", "r"))

model = Seq2Seq(Encoder(len(ichar2ind)), Decoder(len(ochar2ind)))

criterion = nn.CrossEntropyLoss(ignore_index = len(ochar2ind) - 1)
optimizer = optim.Adam(model.parameters(), lr=0.01)
batch_size = 40

n_batches = math.ceil(len(X_train) / batch_size)

#model.load_state_dict(torch.load("C:\CourseWork\Term5\POS_3.pt"))   

for epoch in range(20):
    total_loss = 0
    total_correct = 0
    total = 0
    with tqdm(total = n_batches) as progress_bar:
        for i, (X_batch, y_batch) in enumerate(batches((X_train, y_train), batch_size, len(ochar2ind))):
 
            X_batch, y_batch = LongTensor(X_batch), LongTensor(y_batch)
            res = model(X_batch, y_batch, len(ochar2ind), 1)
            
            #lOSS
            optimizer.zero_grad()
            
            form_res = res[1:].view(-1, res.shape[-1])
            form_y_batch = y_batch[1:].view(-1)
            
            loss = criterion(form_res, form_y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            total_loss += loss.item()
            optimizer.step()
            
            #ACCURACY
            count = 0
            ind = []
            for i in range(0, len(res)):
                l_ind = []
                for j in range(0, len(res[i])):
                    if (y_batch[i][j] != len(ochar2ind) - 1 and y_batch[i][j] != 0):
                        index = torch.argmax(res[i][j])
                    else:
                        index = y_batch[i][j]
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
    
    if epoch == 0 or (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), "C:\CourseWork\Term5\EXP_" + str(epoch) + ".pt")
    progress_bar.set_description('{:>5s} Loss = {:.5f}, Accuracy = {:.2%}'.format
    ("Epoch: ", total_loss / n_batches, float(total_correct) / total))