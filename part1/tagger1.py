import os
import sys
from collections import Counter
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt


TASK = str(sys.argv[1])
TRAIN_BATCH_SIZE = 32
SEPARATOR, DEV_BATCH_SIZE, EPOCHS, IS_NER = (' ', 32, 7, False) if TASK == 'pos' else ('\t', 128, 8, True)


class MlpTagger(nn.Module):

    def __init__(self, vocab_size, hidden_layer_size, output_size, embedding_dim=50, window_size=5):
        super(MlpTagger, self).__init__()

        # Embedding matrix - random initialization
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Linear layers.
        self.linear1 = nn.Linear(embedding_dim * window_size, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, output_size)

        # Defining the non linear activation function to be TanH.
        self.activation = nn.Tanh()

    def forward(self, x):

        # First find the corresponding embeddings vectors and then concatenate them into one long vector.
        x = self.embedding(x).view(-1, 250)

        # For the first linear layer.
        x = self.activation(self.linear1(x))
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
        for batch_idx, (x, y) in enumerate(train_loader):

            # Reset the gradients from the previous iteration.
            optimizer.zero_grad()

            # Prepare the input to the model.
            x = Variable(torch.LongTensor(x))

            # Calculating the model's predictions to the examples in the current batch.
            outputs = model(x)

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
        for batch_idx, (x, y) in enumerate(data_set):

            # Prepare the input to the model.
            x = Variable(torch.LongTensor(x))

            # Calculating the model's predictions to the examples in the current batch.
            outputs = model(x)

            # Get the indexes of the max log-probability.
            prediction = np.argmax(outputs.data.numpy(), axis=1)

            # Calculating the negative log likelihood loss.
            loss = F.nll_loss(outputs, y)
            sum_loss += loss.item()

            # For the ner data set.
            if IS_NER:

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
    sentences, tags = [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence, sentence_tags = [], []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':
                word, tag = line.strip().split(SEPARATOR)
                sentence.append(word)
                sentence_tags.append(tag)

            else:  # Otherwise.
                sentence = [start_token, start_token] + sentence + [end_token, end_token]
                sentences.append(sentence)
                tags.append(sentence_tags)
                sentence, sentence_tags = [], []

    return sentences, tags


# Reading the data from the requested file.
def read_test_data(file_name, start_token='<s>', end_token='<e>'):
    sentences, original = [], []

    with open(file_name, "r", encoding="utf-8") as file:
        data = file.readlines()
        sentence = []

        # For each line in the file.
        for line in data:

            # As long as the sentence isn't over.
            if line is not '\n':
                word = line.split()[0]
                sentence.append(word)

            else:  # Otherwise.
                original.append(sentence)
                sentence = [start_token, start_token] + sentence + [end_token, end_token]
                sentences.append(sentence)
                sentence = []

    return sentences, original


# Considerate rare words like if the were unknown words in order to train the corresponding embedding vector.
def convert_rare_words_to_unknown_token(data, num_occurrences=1, unknown_token='<unk>'):
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


# Making a words vocabulary and tags vocabulary where each word, tag in the training set has a unique index.
def create_words_vocabulary_and_tags_vocabulary(data, tags, unknown_token='<unk>'):
    vocab_words, vocab_tags = [], []

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

    # Go over each sentence in the training set.
    for sentence_tags in tags:

        # For each tag which belongs to a word in the sentence
        for tag in sentence_tags:
            vocab_tags.append(tag)

    # Map each tag to a unique index.
    tag_to_ix = {tag: i for i, tag in enumerate(list(sorted(set(vocab_tags))))}
    ix_to_tag = {i: tag for i, tag in enumerate(list(sorted(set(vocab_tags))))}

    return word_to_ix, ix_to_word, tag_to_ix, ix_to_tag


# Replace each word in the data set with its corresponding index.
def convert_data_to_indexes(data, vocab, unknown_token='<unk>'):
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
def update_data_set_to_window_based_form(data, tags, offset=2):
    windows, targets = [], []

    # For each sentence of words and tags
    for sentence, tags in zip(data, tags):

        # Create window to each word in the sentence.
        contexts = [([sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1], sentence[i + 2]]) for i in
                    range(offset, len(sentence) - offset)]

        # Collecting all the windows in the data set into one list.
        windows += contexts

        # Collecting all the tags in the data set into one list.
        targets += tags

    # Convert the lists into an arrays.
    windows, targets = np.array(windows), np.array(targets)

    return windows, targets


# Update the test set into window based form.
def update_test_set_to_window_based_form(data, offset=2):
    windows = []

    # For each sentence in the test set
    for sentence in data:

        # Create window to each word in the sentence.
        contexts = [([sentence[i - 2], sentence[i - 1], sentence[i], sentence[i + 1], sentence[i + 2]]) for i in
                    range(offset, len(sentence) - offset)]

        # Keep the windows(which represent the words) in the test set in sentences order.
        windows.append(contexts)

    return windows


# Generate Graphs for the task
def generate_graph(folder, y, name_plot, name):
    figure = plt.figure()
    figure.set_size_inches(8, 6)
    plt.plot(range(len(y)), y, linewidth=2.0)
    plt.xlabel("Epochs")
    plt.ylabel(name)
    plt.title(name + " vs Epochs")
    plt.savefig(folder + '/' + name_plot + ".png", dpi=192)


def test_predictions(processed_data, original, index_to_tag):

    # Clearing the content of the file if it already exists; Otherwise, creating the file.
    if os.path.exists("./test1." + TASK):
        os.remove("./test1." + TASK)
    f = open("./test1." + TASK, "a+")

    with torch.no_grad():

        # For each processed sentence in test set and the original (not processed) sentence
        for x, sentence in zip(processed_data, original):

            # Prepare the processed sentence to be the input to the model.
            x = Variable(torch.LongTensor(x))

            # Calculating the model's predictions to the examples in the current sentence.
            outputs = model(x)

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


if __name__ == '__main__':

    # Create a dir in the current working directory in which the generated graphs will be saved.
    output_dir = TASK + "_output_part_1"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    """ Handling the training set """

    # Loading the training set.
    train_data, train_tags = read_data("./" + TASK + "/train")

    # Consider rare words as unknown words.
    train_data = convert_rare_words_to_unknown_token(train_data)

    # Assigning a unique index to each word, tag in the training set.
    w2i, i2w, t2i, i2t = create_words_vocabulary_and_tags_vocabulary(train_data, train_tags)

    # Update to indexes representation.
    train_data = convert_data_to_indexes(train_data, w2i)
    train_tags = convert_tags_to_indexes(train_tags, t2i)

    # Organize the data set into context windows.
    train_contexts, train_tags = update_data_set_to_window_based_form(train_data, train_tags)

    # Creating a torch loader.
    train_data_set = TensorDataset(torch.LongTensor(train_contexts), torch.LongTensor(train_tags))
    train_loader = DataLoader(train_data_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True)

    """ Handling the dev set """

    # Loading the dev set.
    dev_data, dev_tags = read_data("./" + TASK + "/dev")

    # Update to indexes representation.
    dev_data = convert_data_to_indexes(dev_data, w2i)
    dev_tags = convert_tags_to_indexes(dev_tags, t2i)

    # Organize the data set into context windows.
    dev_contexts, dev_tags = update_data_set_to_window_based_form(dev_data, dev_tags)

    # Creating a torch loader.
    dev_data_set = TensorDataset(torch.LongTensor(dev_contexts), torch.LongTensor(dev_tags))
    dev_loader = DataLoader(dev_data_set, batch_size=DEV_BATCH_SIZE, shuffle=True)

    """ Handling the test set """

    # Loading the test set.
    test_data, original_data = read_test_data("./" + TASK + "/test")

    # Update to indexes representation.
    test_data = convert_data_to_indexes(test_data, w2i)

    # Organize the data set into context windows.
    test_contexts = update_test_set_to_window_based_form(test_data)

    # Creating an instance of MlpTagger.
    model = MlpTagger(vocab_size=len(w2i), hidden_layer_size=150, output_size=len(t2i))

    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Training the model
    epochs_Dev_Loss, epochs_Dev_Acc = train(model, optimizer, train_loader, dev_loader, EPOCHS, i2t)

    # Generate graphs describing the dev's loss and accuracy as a function on the number of epochs.
    generate_graph(output_dir, epochs_Dev_Loss, "dev_loss", "Dev Loss")
    generate_graph(output_dir, epochs_Dev_Acc, "dev_accuracy", "Dev Accuracy")

    # Calculating the predictions of the model to the test examples and writing each one to the file.
    test_predictions(test_contexts, original_data, i2t)
