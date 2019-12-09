# CourseProject1
Research of sequence generation methods by using recurrent neural networks

Monodirectional RNN model:
  - source code: POS_TrainMode.py (code for training model), POS_TestMode.py (code for testing model)
  - pretrained models: POS_N.pt (N in [0, 3]) - preatrained model after (N + 1) epochs

Bidirectional RNN model:
1) Large bididrectional RNN model (hidden dimension = 128):
  - source code: POS_TrainMode_Bi.py (code for training model), POS_TestMode_Bi.py (code for testing model)
  - pretrained models: POS_BiN.pt (N in [1, 3]) - preatrained model after (N + 1) epochs
2) Small bidirectional RNN model (hidden dimension = 64):
  - source code: just change the hidden dimension value on 64 in POS_TrainMode_Bi.py and POS_TestMode_Bi.py
  - pretrained models: POS_Bi_N.pt (N in [0, 8]) - prettrained model after (N + 1) epochs

Seq2Seq model:
- source code EXP_TrainMode.py

Dataset for task Part Of Speech Tagging:
- training data:
  - original: test.txt
  - prepared: test_data.txt
- validation data
  - original: val.txt
  - prepared: val_data.txt
- testing data
  - original: test.txt
  - prepared: test_data.txt

Other input data:
- dictionaries
  - word2ind (from word to integer number)
  - tag2ind (from tag to integer number)
- pretrained embeddings (link - https://drive.google.com/open?id=1Bk5JdDX2ojd_6JF8a45-Pu3QO9D7tQzZ) // file is too large

Not-NN solution of POS problem:
- Ngram_Tagger.py

Just run TestMode files to test preatrained models (don't forget to change path of current directory in os.chdir() method).

File Statistics.xlsx contains statistics data used in article
