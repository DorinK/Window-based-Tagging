README for tagger2:

In the current working directory should be:
1.tagger2.py
2.wordVectors	(txt file contains the embedding vectors)
3.vocab	(txt file contains the vocabulary words)
4.pos	(directory)
5.ner	(directory)

In the pos directory should be the corresponding train, dev and test files.
In the ner directory should be the corresponding train, dev and test files.

One parameter to the main - the task you want(pos / ner).

From the command line you should enter: 
python3 tagger2.py pos
python3 tagger2.py ner

The output graphs will be in the respective folders:
pos_output_part_3
ner_output_part_3