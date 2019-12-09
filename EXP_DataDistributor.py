import random

import json
import os

os.chdir("C:\CourseWork\Term5")

def convert_data(data, ichar2ind, ochar2ind):
    X = [[0] + [ichar2ind[expc] for expc in sample[0]] + [1] for sample in data]
    y = [[0] + [ochar2ind[resc] for resc in sample[1]] + [1] for sample in data]
        
    return X, y

ichar = ['<sos>', '<eos>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '<pad>']
ichar2ind = {char: ind for ind, char in enumerate(ichar)}
json.dump(ichar2ind, open("EXP_ichar2ind.txt", "w+"))

ochar = ['<sos>', '<eos>', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '<pad>']
ochar2ind = {char: ind for ind, char in enumerate(ochar)}
json.dump(ochar2ind, open("EXP_ochar2ind.txt", "w+"))

data = []
for i in range(50000):
    first = random.randint(-1e4, 1e4)
    second = random.randint(-1e4, 1e4)
    exp = str(first) + ("+" if second >= 0 else "") + str(second)
    res = first + second
    
    data.append([[c for c in exp],[c for c in str(res)]])
    
json.dump(data, open("EXP_data.txt", "w+"))

train_data = data[:30000]
val_data = data[30000:37500]
test_data = data[37500:]

json.dump(train_data, open("EXP_train.txt", "w+"))
json.dump(val_data, open("EXP_val.txt", "w+"))
json.dump(test_data, open("EXP_test.txt", "w+"))


X_train, y_train = convert_data(train_data, ichar2ind, ochar2ind)
X_val, y_val = convert_data(val_data, ichar2ind, ochar2ind)
X_test, y_test = convert_data(test_data, ichar2ind, ochar2ind)

json.dump((X_train, y_train), open("EXP_train_data.txt", "w+"))
json.dump((X_val, y_val), open("EXP_val_data.txt", "w+"))
json.dump((X_test, y_test), open("EXP_test_data.txt", "w+"))