import tagger3_pre_trained
import tagger3_random
import sys


TASK = str(sys.argv[1])
CONDITION = str(sys.argv[2])

SEPARATOR, BATCH_SIZE, EPOCHS, IS_NER = (
    (' ', 128, 6, False) if TASK == 'pos' else ('\t', 16, 7, True)) if CONDITION == 'random' else (
    (' ', 32, 5, False) if TASK == 'pos' else ('\t', 16, 6, True))

if CONDITION == "pre-trained":
    tagger3_pre_trained.main()

elif CONDITION == "random":
    tagger3_random.main()
