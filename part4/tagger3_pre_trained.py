import os
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import tagger3


def check_if_a_number(word, vocab):

    # If a number is of pattern 'DGDG', 'DG.DG', '.DG', '+DG', '-DG' and etc.
    if all(ch.isdigit() or ch == '.' or ch == '+' or ch == '-' for ch in word):
        pattern = ""

        # Replace each character with 'DG'
        for ch in word:
            pattern += 'DG' if ch.isdigit() else ch

        # If this pattern is in the pre-trained vocabulary return it; Otherwise, return the pattern 'NNNUMMM'
        pattern = pattern if pattern in vocab else 'NNNUMMM'
        return pattern

    # If a number is of pattern '_ ,_ _' ; '_ ,_ _ _, _ _ _' and etc return the pattern 'NNNUMMM'.
    elif all(ch.isdigit() or ch == ',' for ch in word) and any(ch.isdigit() for ch in word):
        return "NNNUMMM"

    return None


class MlpTagger(nn.Module):

    def __init__(self, words_vocab_size, prefix_vocab_size, suffix_vocab_size, embeddings, hidden_layer_size,
                 output_size, embedding_dim=50, window_size=5):
        super(MlpTagger, self).__init__()

        # 3 Embedding matrices - for the words vocabulary, for the prefixes vocabulary and for the suffixes vocabulary.
        self.embedding = nn.Embedding(words_vocab_size, embedding_dim)
        self.prefix_embeddings = nn.Embedding(prefix_vocab_size, embedding_dim)
        self.suffix_embeddings = nn.Embedding(suffix_vocab_size, embedding_dim)

        # The words embedding matrix will contain the pre trained embeddings.
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

        # Linear layers.
        self.linear1 = nn.Linear(embedding_dim * window_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        # Defining the non linear activation function to be TanH.
        self.activation = nn.Tanh()

    def forward(self, x, prefixes, suffixes):

        # Finding the corresponding embeddings vectors of the prefixes and then concatenate them into one long vector.
        prefixes = self.prefix_embeddings(prefixes).view(-1, 250)

        # Finding the corresponding embeddings vectors of the words and then concatenate them into one long vector.
        x = self.embedding(x).view(-1, 250)

        # Finding the corresponding embeddings vectors of the suffixes and then concatenate them into one long vector.
        suffixes = self.prefix_embeddings(suffixes).view(-1, 250)

        # For the first linear layer - the input is the sum.
        x = self.activation(self.linear1(prefixes + x + suffixes))
        x = F.dropout(x, training=self.training)  # Done to prevent over fitting.

        # For the second linear layer.
        x = self.linear2(x)

        return F.log_softmax(x, dim=-1)


def train(model, optimizer, train_loader, dev_loader, epochs, index_to_tag):

    # Lists that will contain the loss and accuracy of the dev set in each epoch.
    epochs_dev_acc, epochs_dev_loss = [], []

    for epoch in range(epochs):

        # Declaring training mode.
        model.train()

        sum_loss = 0.0
        for batch_idx, (x, pref_x, suf_x, y) in enumerate(train_loader):

            # Reset the gradients from the previous iteration.
            optimizer.zero_grad()

            # Prepare the inputs to the model.
            x, pref_x, suf_x = Variable(torch.LongTensor(x)), Variable(torch.LongTensor(pref_x)), Variable(
                torch.LongTensor(suf_x))

            # Calculating the model's predictions to the examples in the current batch.
            outputs = model(x, pref_x, suf_x)

            # Calculating the negative log likelihood loss.
            loss = F.nll_loss(outputs, y)
            sum_loss += loss.item()

            # Back propagation.
            loss.backward()

            # Updating.
            optimizer.step()

        # Calculating the loss on the training set in the current epoch.
        train_loss = sum_loss / len(train_loader.dataset)

        # Calculating the accuracy on the training set in the current epoch.
        train_accuracy, _ = accuracy_on_data_set(model, train_loader, index_to_tag)

        # Calculating the loss and accuracy on the dev set in the current epoch.
        dev_accuracy, dev_loss = accuracy_on_data_set(model, dev_loader, index_to_tag)

        # Save the dev's loss and accuracy results.
        epochs_dev_loss.append(dev_loss)
        epochs_dev_acc.append(dev_accuracy)

        print(epoch, train_loss, train_accuracy, dev_loss, dev_accuracy)

    return epochs_dev_loss, epochs_dev_acc


def accuracy_on_data_set(model, data_set, index_to_tag):

    # Declaring evaluation mode.
    model.eval()

    good = total = 0.0
    sum_loss = 0.0

    with torch.no_grad():
        for batch_idx, (x, pref_x, suf_x, y) in enumerate(data_set):

            # Prepare the inputs to the model.
            x, pref_x, suf_x = Variable(torch.LongTensor(x)), Variable(torch.LongTensor(pref_x)), Variable(
                torch.LongTensor(suf_x))

            # Calculating the model's predictions to the examples in the current batch.
            outputs = model(x, pref_x, suf_x)

            # Get the indexes of the max log-probability.
            prediction = np.argmax(outputs.data.numpy(), axis=1)

            # Calculating the negative log likelihood loss.
            loss = F.nll_loss(outputs, y)
            sum_loss += loss.item()

            # For the ner data set.
            if tagger3.IS_NER:

                # For each prediction and tag of an example in the batch
                for y_hat, tag in np.nditer([prediction, y.numpy()]):
                    total += 1

                    # Don't count the cases in which both prediction and tag are 'O'.
                    if y_hat == tag:
                        if index_to_tag[int(y_hat)] == 'O':
                            total -= 1
                        else:
                            good += 1

            else:  # For the pos data set.
                res = (prediction == y.numpy())
                total += len(y)
                good += np.sum(res)

    # Calculating the loss and accuracy rate on the data set.
    return good / total, sum_loss / len(data_set.dataset)


# Reading the data from the requested file.
def read_data(file_name, start_token='<s>', end_token='</s>'):
    sentences, tags, prefixes, suffixes = [], [], [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':

                word, tag = line.strip().split(tagger3.SEPARATOR)
                sentence.append(word)
                sentence_tags.append(tag)

                # For each word save it's prefix and suffix.
                sentence_prefixes.append(word[:3])
                sentence_suffixes.append(word[-3:])

            else:  # Otherwise.
                sentence = [start_token, start_token] + sentence + [end_token, end_token]
                sentence_prefixes = [start_token[:3], start_token[:3]] + sentence_prefixes + [end_token[:3],
                                                                                              end_token[:3]]
                sentence_suffixes = [start_token[-3:], start_token[-3:]] + sentence_suffixes + [end_token[-3:],
                                                                                                end_token[-3:]]
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                tags.append(sentence_tags)
                sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

    return sentences, prefixes, suffixes, tags


# Reading the data from the requested file.
def read_test_data(file_name, start_token='<s>', end_token='</s>'):
    sentences, original, prefixes, suffixes = [], [], [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_prefixes, sentence_suffixes = [], [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':

                word = line.split()[0]
                sentence.append(word)

                # For each word save it's prefix and suffix.
                sentence_prefixes.append(word[:3])
                sentence_suffixes.append(word[-3:])

            else:  # Otherwise.
                original.append(sentence)
                sentence = [start_token, start_token] + sentence + [end_token, end_token]
                sentence_prefixes = [start_token[:3], start_token[:3]] + sentence_prefixes + [end_token[:3],
                                                                                              end_token[:3]]
                sentence_suffixes = [start_token[-3:], start_token[-3:]] + sentence_suffixes + [end_token[-3:],
                                                                                                end_token[-3:]]
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                sentence, sentence_prefixes, sentence_suffixes = [], [], []

    return sentences, original, prefixes, suffixes


# Making a words vocabulary where each word has a unique index.
def create_words_vocabulary(data):

    # Map each word to a unique index.
    word_to_ix = {word: i for i, word in enumerate(list(sorted(set(data))))}
    ix_to_word = {i: word for i, word in enumerate(list(sorted(set(data))))}
    return word_to_ix, ix_to_word


# Making a tags vocabulary where each tag in the training set has a unique index.
def create_tags_vocabulary(tags):
    vocab_tags = []

    # Go over each sentence in the training set.
    for sentence_tags in tags:

        # For each tag which belongs to a word in the sentence
        for tag in sentence_tags:
            vocab_tags.append(tag)

    # Map each tag to a unique index.
    tag_to_ix = {tag: i for i, tag in enumerate(list(sorted(set(vocab_tags))))}
    ix_to_tag = {i: tag for i, tag in enumerate(list(sorted(set(vocab_tags))))}

    return tag_to_ix, ix_to_tag


# Replace each word, prefix, suffix in the data set with its corresponding index.
def convert_data_to_indexes(data, prefixes, suffixes, word2i, pref2i, suf2i, unknown_token='UUUNKKK'):
    word_sentences, words_indexes, pref_sentences, prefix_indexes, suf_sentences, suffix_indexes = [], [], [], [], [], []

    # For words, prefixes and suffixes of each sentence
    for words_sentence, prefix_sentence, suffix_sentence in zip(data, prefixes, suffixes):

        # For each word in the sentence and it's prefix and suffix
        for word, prefix, suffix in zip(words_sentence, prefix_sentence, suffix_sentence):

            #  If the word is in the vocabulary.
            if word in word2i:
                ix = word2i.get(word)  # Get its corresponding index.
                pref_ix = pref2i.get(prefix)  # Get the word's prefix corresponding index.
                suf_ix = suf2i.get(suffix)  # Get the word's suffix corresponding index.

            # If not, check for existence of the word with lower letters.
            elif word.lower() in word2i:
                ix = word2i.get(word.lower())
                pref_ix = pref2i.get(prefix.lower())
                suf_ix = suf2i.get(suffix.lower())

            # If not, check if the word is a number.
            elif check_if_a_number(word, word2i) in word2i:
                pattern = check_if_a_number(word, word2i)
                ix = word2i.get(pattern)
                pref_ix = pref2i.get(pattern[:3])
                suf_ix = suf2i.get(pattern[-3:])

            else:
                # Otherwise, assign to it the index of the unknown token.
                ix = word2i.get(unknown_token)

                # If the prefix / suffix of the word is in the prefix / suffix vocabulary get its corresponding index;
                # Otherwise, assign to it the prefix / suffix of the unknown token.
                pref_ix = pref2i.get(prefix.lower()) if prefix.lower() in pref2i else pref2i.get(unknown_token[:3])
                suf_ix = suf2i.get(suffix.lower()) if suffix.lower() in suf2i else suf2i.get(unknown_token[-3:])

            words_indexes.append(ix)
            prefix_indexes.append(pref_ix)
            suffix_indexes.append(suf_ix)

        # Keep the words, prefixes and suffixes in the data set in sentences order.
        word_sentences.append(words_indexes)
        pref_sentences.append(prefix_indexes)
        suf_sentences.append(suffix_indexes)

        words_indexes, prefix_indexes, suffix_indexes = [], [], []

    return word_sentences, pref_sentences, suf_sentences


# Replace each tag of a word in the data set with its corresponding index.
def convert_tags_to_indexes(tags, vocab):
    sentences, tags_indexes = [], []

    # Go over each sentence in the training set.
    for sentence in tags:

        # For each tag of a word in the sentence
        for tag_of_word in sentence:

            # Find its corresponding index
            ix = vocab.get(tag_of_word)
            tags_indexes.append(ix)

        # Keep the tags in the data set in sentences order.
        sentences.append(tags_indexes)
        tags_indexes = []

    # Return the updated tags data
    return sentences


# Update the data set (training set / dev set) into window based form.
def update_data_set_to_window_based(data, prefixes, suffixes, tags, offset=2):
    words_windows, prefixes_windows, suffixes_windows, targets = [], [], [], []

    # For words, prefixes, suffixes and tags of each sentence
    for sentence, sentence_prefixes, sentence_suffixes, sentence_tags in zip(data, prefixes, suffixes, tags):

        # Create context window to each word in the sentence.
        contexts = [([sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1], sentence[i + 2]]) for i in
                    range(offset, len(sentence) - offset)]

        # Organize the prefixes in the sentence in the same pattern as before.
        pref = [([sentence_prefixes[i - 2], sentence_prefixes[i - 1], sentence_prefixes[i], sentence_prefixes[i + 1],
                  sentence_prefixes[i + 2]]) for i in range(offset, len(sentence_prefixes) - offset)]

        # Organize the suffixes in the sentence in the same pattern as before.
        suff = [([sentence_suffixes[i - 2], sentence_suffixes[i - 1], sentence_suffixes[i], sentence_suffixes[i + 1],
                  sentence_suffixes[i + 2]]) for i in range(offset, len(sentence_suffixes) - offset)]

        # Collecting all the context windows of words in the data set into one list.
        words_windows += contexts

        # Collecting all the context windows of prefixes in the data set into one list.
        prefixes_windows += pref

        # Collecting all the context windows of suffixes in the data set into one list.
        suffixes_windows += suff

        # Collecting all the context windows of tags in the data set into one list.
        targets += sentence_tags

    # Convert the lists into an arrays.
    words_windows, prefixes_windows, suffixes_windows, targets = np.array(words_windows), np.array(
        prefixes_windows), np.array(suffixes_windows), np.array(targets)

    return words_windows, prefixes_windows, suffixes_windows, targets


# Update the test set into window based form.
def update_test_set_to_window_based(data, prefixes, suffixes, offset=2):
    words_windows, prefixes_windows, suffixes_windows = [], [], []

    # For words, prefixes and suffixes of each sentence
    for sentence, sentence_prefixes, sentence_suffixes in zip(data, prefixes, suffixes):

        # Create context window to each word in the sentence.
        contexts = [([sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1], sentence[i + 2]]) for i in
                    range(offset, len(sentence) - offset)]

        # Organize the prefixes in the sentence in the same pattern as before.
        pref = [([sentence_prefixes[i - 2], sentence_prefixes[i - 1], sentence_prefixes[i], sentence_prefixes[i + 1],
                  sentence_prefixes[i + 2]]) for i in range(offset, len(sentence_prefixes) - offset)]

        # Organize the suffixes in the sentence in the same pattern as before.
        suff = [([sentence_suffixes[i - 2], sentence_suffixes[i - 1], sentence_suffixes[i], sentence_suffixes[i + 1],
                  sentence_suffixes[i + 2]]) for i in range(offset, len(sentence_suffixes) - offset)]

        # Keep the context windows in the test set in sentences order.
        words_windows.append(contexts)
        prefixes_windows.append(pref)
        suffixes_windows.append(suff)

    return words_windows, prefixes_windows, suffixes_windows


# Generate Graphs for the task.
def generate_graph(folder, y, name_plot, name):
    figure = plt.figure()
    figure.set_size_inches(8, 6)
    plt.plot(range(len(y)), y, linewidth=2.0)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name + " vs Epochs")
    plt.savefig(folder + '/' + name_plot + ".png", dpi=192)


def test_predictions(model, processed_data, prefixes, suffixes, original, index_to_tag):

    # Clearing the content of the file if it already exists; Otherwise, creating the file.
    if os.path.exists("./test4." + tagger3.TASK):
        os.remove("./test4." + tagger3.TASK)
    f = open("./test4." + tagger3.TASK, "a+")

    with torch.no_grad():

        # For each processed sentence (data,prefixes,suffixes) in the test set and the original(not processed) sentence.
        for x, pref_x, suf_x, sentence in zip(processed_data, prefixes, suffixes, original):

            # Prepare the inputs to the model.
            x, pref_x, suf_x = Variable(torch.LongTensor(x)), Variable(torch.LongTensor(pref_x)), Variable(
                torch.LongTensor(suf_x))

            # Calculating the model's predictions to the examples in the current sentence.
            outputs = model(x, pref_x, suf_x)

            # Get the index of the max log-probability.
            predictions = np.argmax(outputs.data.numpy(), axis=1)

            # For each word in the current original sentence and its corresponding prediction
            for word, prediction in zip(sentence, list(predictions)):

                #  Write to the file.
                f.write("{0} {1}\n".format(word, index_to_tag[prediction]))

            # Add new line after each sentence (following the requested format).
            f.write("\n")

    # Close the file.
    f.close()


# Saving the prefixes and suffixes of the words in the pre-trained vocabulary.
def find_prefixes_and_suffixes(vocab):
    prefixes, suffixes = [], []

    # For each word in the pre-trained vocabulary keep it's prefix and suffix.
    for word in vocab:
        prefixes.append(word[:3])
        suffixes.append(word[-3:])

    return prefixes, suffixes


def main():

    # Create a dir in the current working directory in which the generated graphs will be saved.
    output_dir = tagger3.TASK + "_output_part_4_pre_trained"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading the pre-trained embeddings.
    vecs = np.loadtxt("wordVectors.txt")

    # Loading the pre-trained vocabulary.
    with open("vocab.txt", "r", encoding="utf-8") as file:
        words_vocab = file.readlines()
        words_vocab = [word.strip() for word in words_vocab]

    # Saving the prefixes and suffixes of the words in the pre-trained vocabulary.
    prefix_vocab, suffix_vocab = find_prefixes_and_suffixes(words_vocab)

    # Assigning a unique index to each word, prefix, suffix in the pre-trained vocabulary.
    w2i, i2w = create_words_vocabulary(words_vocab)
    p2i, i2p = create_words_vocabulary(prefix_vocab)
    s2i, i2s = create_words_vocabulary(suffix_vocab)

    """ Handling the training set """

    # Loading the training set.
    train_data, train_prefixes, train_suffixes, train_tags = read_data("./" + tagger3.TASK + "/train")

    # Assigning a unique index to each tag in the training set.
    t2i, i2t = create_tags_vocabulary(train_tags)

    # Update to indexes representation.
    train_data, train_prefixes, train_suffixes = convert_data_to_indexes(train_data, train_prefixes, train_suffixes,
                                                                         w2i, p2i, s2i)
    train_tags = convert_tags_to_indexes(train_tags, t2i)

    # Organize the data set into context windows.
    train_words_contexts, train_prefix_contexts, train_suffix_contexts, train_tags = update_data_set_to_window_based(
        train_data, train_prefixes, train_suffixes, train_tags)

    # Creating a torch loader.
    train_data_set = TensorDataset(torch.LongTensor(train_words_contexts), torch.LongTensor(train_prefix_contexts),
                                   torch.LongTensor(train_suffix_contexts), torch.LongTensor(train_tags))
    train_loader = DataLoader(train_data_set, batch_size=tagger3.BATCH_SIZE, shuffle=True)

    """ Handling the dev set """

    # Loading the dev set.
    dev_data, dev_prefixes, dev_suffixes, dev_tags = read_data("./" + tagger3.TASK + "/dev")

    # Update to indexes representation.
    dev_data, dev_prefixes, dev_suffixes = convert_data_to_indexes(dev_data, dev_prefixes, dev_suffixes, w2i, p2i, s2i)
    dev_tags = convert_tags_to_indexes(dev_tags, t2i)

    # Organize the data set into context windows.
    dev_words_contexts, dev_prefix_contexts, dev_suffix_contexts, dev_tags = update_data_set_to_window_based(
        dev_data, dev_prefixes, dev_suffixes, dev_tags)

    # Creating a torch loader.
    dev_data_set = TensorDataset(torch.LongTensor(dev_words_contexts), torch.LongTensor(dev_prefix_contexts),
                                 torch.LongTensor(dev_suffix_contexts), torch.LongTensor(dev_tags))
    dev_loader = DataLoader(dev_data_set, batch_size=tagger3.BATCH_SIZE, shuffle=True)

    """ Handling the test set """

    # Loading the test set.
    test_data, original_data, test_prefixes, test_suffixes = read_test_data("./" + tagger3.TASK + "/test")

    # Update to indexes representation.
    test_data, test_prefixes, test_suffixes = convert_data_to_indexes(test_data, test_prefixes, test_suffixes, w2i, p2i,
                                                                      s2i)

    # Organize the data set into context windows.
    test_words_contexts, test_prefix_contexts, test_suffix_contexts = update_test_set_to_window_based(
        data=test_data, prefixes=test_prefixes, suffixes=test_suffixes)

    # Creating an instance of MlpTagger.
    model = MlpTagger(words_vocab_size=len(w2i), prefix_vocab_size=len(p2i), suffix_vocab_size=len(s2i),
                      embeddings=vecs, hidden_layer_size=150, output_size=len(t2i))

    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Training the model
    epochs_Dev_Loss, epochs_Dev_Acc = train(model, optimizer, train_loader, dev_loader, tagger3.EPOCHS, i2t)

    # Generate graphs describing the dev's loss and accuracy as a function on the number of epochs.
    generate_graph(output_dir, epochs_Dev_Loss, "dev_loss", "Dev Loss")
    generate_graph(output_dir, epochs_Dev_Acc, "dev_accuracy", "Dev Accuracy")

    # Calculating the predictions of the model to the test examples and writing each one to the file.
    test_predictions(model, test_words_contexts, test_prefix_contexts, test_suffix_contexts, original_data, i2t)


if __name__ == '__main__':
    main()
