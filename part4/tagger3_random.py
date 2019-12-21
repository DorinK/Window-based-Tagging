import os
from collections import Counter
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import tagger3


class MlpTagger(nn.Module):

    def __init__(self, words_vocab_size, prefix_vocab_size, suffix_vocab_size, hidden_layer_size, output_size,
                 embedding_dim=50, window_size=5):
        super(MlpTagger, self).__init__()

        # 3 Embedding matrices - for the words vocabulary, for the prefixes vocabulary and for the suffixes vocabulary.
        self.embedding = nn.Embedding(words_vocab_size, embedding_dim)  # random initialization
        self.prefix_embeddings = nn.Embedding(prefix_vocab_size, embedding_dim)  # random initialization
        self.suffix_embeddings = nn.Embedding(suffix_vocab_size, embedding_dim)  # random initialization

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
def read_data(file_name, start_token='<s>', end_token='<e>'):
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
                sentence_prefixes = [start_token, start_token] + sentence_prefixes + [end_token, end_token]
                sentence_suffixes = [start_token, start_token] + sentence_suffixes + [end_token, end_token]
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                tags.append(sentence_tags)
                sentence, sentence_tags, sentence_prefixes, sentence_suffixes = [], [], [], []

    return sentences, prefixes, suffixes, tags


# Reading the data from the requested file.
def read_test_data(file_name, start_token='<s>', end_token='<e>'):
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
                sentence_prefixes = [start_token, start_token] + sentence_prefixes + [end_token, end_token]
                sentence_suffixes = [start_token, start_token] + sentence_suffixes + [end_token, end_token]
                sentences.append(sentence)
                prefixes.append(sentence_prefixes)
                suffixes.append(sentence_suffixes)
                sentence, sentence_prefixes, sentence_suffixes = [], [], []

    return sentences, original, prefixes, suffixes


# Considerate rare words like if the were unknown words in order to train the corresponding embedding vector.
def convert_rare_words_to_unknown_token(data, num_occurrences=1, unknown_token='UNK'):
    count = Counter()
    convert_to_unk = set()

    # Count the number of occurrences of each word in the training set.
    for sentence in data:
        count.update(sentence)

    # Collect the words in the training set that appear only once.
    for word, amount in count.items():
        if amount <= num_occurrences:
            convert_to_unk.add(word)

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for i in range(len(sentence)):

            # If the current word appears only once then considerate it as unknown word.
            if sentence[i] in convert_to_unk:
                sentence[i] = unknown_token

    # Return the updated training set data.
    return data


# Making a words vocabulary where each word has a unique index.
def create_words_vocabulary(data, unknown_token='UNK'):
    vocab_words = []

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:
            vocab_words.append(word)

    # Add the unknown_token.
    vocab_words.append(unknown_token)

    # Map each word to a unique index.
    word_to_ix = {word: i for i, word in enumerate(list(sorted(set(vocab_words))))}
    ix_to_word = {i: word for i, word in enumerate(list(sorted(set(vocab_words))))}
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


# Replace each word in the data set with its corresponding index.
def convert_data_to_indexes(data, vocab, unknown_token='UNK'):
    sentences, words_indexes = [], []

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:

            # Find its corresponding index - if not exist then assign the index of the unknown_token.
            ix = vocab.get(word) if word in vocab else vocab.get(unknown_token)
            words_indexes.append(ix)

        # Keep the words in the data set in sentences order.
        sentences.append(words_indexes)
        words_indexes = []

    # Return the updated data
    return sentences


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


# Generate Graphs for the task
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


def main():

    # Create a dir in the current working directory in which the generated graphs will be saved.
    output_dir = tagger3.TASK + "_output_part_4_random"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """ Handling the training set """

    # Loading the training set.
    train_data, train_prefixes, train_suffixes, train_tags = read_data("./" + tagger3.TASK + "/train")

    # Consider rare words as unknown words.
    train_data = convert_rare_words_to_unknown_token(train_data)

    # Consider rare prefixes as unknown prefixes.
    train_prefixes = convert_rare_words_to_unknown_token(train_prefixes)

    # Consider rare suffixes as unknown suffixes.
    train_suffixes = convert_rare_words_to_unknown_token(train_suffixes)

    # Assigning a unique index to each word, prefix, suffix, tag in the vocabulary.
    w2i, i2w = create_words_vocabulary(train_data)
    p2i, i2p = create_words_vocabulary(train_prefixes)
    s2i, i2s = create_words_vocabulary(train_suffixes)
    t2i, i2t = create_tags_vocabulary(train_tags)

    # Update to indexes representation.
    train_data = convert_data_to_indexes(train_data, w2i)
    train_prefixes = convert_data_to_indexes(train_prefixes, p2i)
    train_suffixes = convert_data_to_indexes(train_suffixes, s2i)
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
    dev_data = convert_data_to_indexes(dev_data, w2i)
    dev_prefixes = convert_data_to_indexes(dev_prefixes, p2i)
    dev_suffixes = convert_data_to_indexes(dev_suffixes, s2i)
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
    test_data = convert_data_to_indexes(test_data, w2i)
    test_prefixes = convert_data_to_indexes(test_prefixes, p2i)
    test_suffixes = convert_data_to_indexes(test_suffixes, s2i)

    # Organize the data set into context windows.
    test_words_contexts, test_prefix_contexts, test_suffix_contexts = update_test_set_to_window_based(
        data=test_data, prefixes=test_prefixes, suffixes=test_suffixes)

    # Creating an instance of MlpTagger.
    model = MlpTagger(words_vocab_size=len(w2i), prefix_vocab_size=len(p2i), suffix_vocab_size=len(s2i),
                      hidden_layer_size=150, output_size=len(t2i))

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
