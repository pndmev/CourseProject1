import nltk
import json
import os

def convert_data(data):
    conv_data = []
    for sentence in data:
        conv_sent = []
        for word, tag in sentence:
            conv_sent.append((word, tag))
        conv_data.append(conv_sent)
    return conv_data

os.chdir('C:\CourseWork\Term5')

train = json.load(open('train.txt'))
test = json.load(open('test.txt'))

train_data = convert_data(train)
test_data = convert_data(test)

default_tagger = nltk.DefaultTagger('NN')

unigram_tagger = nltk.NgramTagger(1, train_data, backoff = default_tagger)
unigram_tagger.evaluate(test_data)

bigram_tagger = nltk.NgramTagger(2, train_data, backoff = unigram_tagger)
bigram_tagger.evaluate(test_data)

trigram_tagger = nltk.NgramTagger(3, train_data, backoff = bigram_tagger)
trigram_tagger.evaluate(test_data)

fourgram_tagger = nltk.NgramTagger(4, train_data, backoff = trigram_tagger)
fourgram_tagger.evaluate(test_data)

fivegram_tagger = nltk.NgramTagger(5, train_data, backoff = fourgram_tagger)
fivegram_tagger.evaluate(test_data)