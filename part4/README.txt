README for tagger3:

In the current working directory should be:
1.tagger3.py
2.tagger3_random.py
3.tagger3_pre_trained.py
4.wordVectors	(txt file contains the embedding vectors)
5.vocab	(txt file contains the vocabulary words)
6.pos	(directory)
7.ner	(directory)

In the pos directory should be the corresponding train, dev and test files.
In the ner directory should be the corresponding train, dev and test files.

Two parameters to the main are:
1. The task you want(pos / ner).
2. The condition of the embeddings initialization (pre-trained / random)

From the command line you should enter: 
python3 tagger3.py pos random
python3 tagger3.py pos pre-trained
python3 tagger3.py ner random
python3 tagger3.py ner pre-trained

The output graphs will be in the respective folders:
pos_output_part_4_random
ner_output_part_4_random
pos_output_part_4_pre_trained
ner_output_part_4_pre_trained