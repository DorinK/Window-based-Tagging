import os
import sys
import torch
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt


TASK = str(sys.argv[1])
SEPARATOR, BATCH_SIZE, EPOCHS, IS_NER = (' ', 32, 2, False) if TASK == 'pos' else ('\t', 16, 3, True)


# If a word is a number then trying to check if its pattern is in the pre-trained vocabulary.
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

    def __init__(self, vocab_size, embeddings, hidden_layer_size, output_size, embedding_dim=50, window_size=5):
        super(MlpTagger, self).__init__()

        # Embedding matrix
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # The embedding matrix will contain the pre trained embeddings.
        self.embedding.weight.data.copy_(torch.from_numpy(embeddings))

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
def read_data(file_name, start_token='<s>', end_token='</s>'):
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
def read_test_data(file_name, start_token='<s>', end_token='</s>'):
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


# Replace each word in the data set with its corresponding index.
def convert_data_to_indexes(data, vocab, unknown_token='UUUNKKK'):
    sentences, words_indexes = [], []

    # Go over each sentence in the training set.
    for sentence in data:

        # For each word in the sentence
        for word in sentence:

            # If the current word is in the vocabulary get its corresponding index.
            if word in vocab:
                ix = vocab.get(word)

            # If not, check for existence of the word with lower letters.
            elif word.lower() in vocab:
                ix = vocab.get(word.lower())

            # If not, check if the word is a number.
            elif check_if_a_number(word, vocab) in vocab:
                ix = vocab.get(check_if_a_number(word, vocab))

            else:  # Otherwise, assign to it the index of the unknown token.
                ix = vocab.get(unknown_token)
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
def update_data_set_to_window_based(data, tags, offset=2):
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
def update_test_set_to_window_based(data, offset=2):
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
    if os.path.exists("./test3." + TASK):
        os.remove("./test3." + TASK)
    file = open("./test3." + TASK, "a+")

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
                file.write("{0} {1}\n".format(word, index_to_tag[prediction]))

            # Add new line after each sentence (following the requested format).
            file.write("\n")

    # Close the file.
    file.close()


if __name__ == '__main__':

    # Create a dir in the current working directory in which the generated graphs will be saved.
    output_dir = TASK + "_output_part_3"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loading the pre-trained embeddings.
    vecs = np.loadtxt("./wordVectors.txt")

    # Loading the pre-trained vocabulary.
    with open("./vocab.txt", "r", encoding="utf-8") as f:
        vocabulary = f.readlines()
        vocabulary = [word.strip() for word in vocabulary]

    # Assigning an index to each word in the pre-trained vocabulary.
    w2i, i2w = create_words_vocabulary(vocabulary)

    """ Handling the training set """

    # Loading the data sets of each section respectively.
    train_data, train_tags = read_data("./" + TASK + "/train")

    # Assigning a unique index to each tag in the training set.
    t2i, i2t = create_tags_vocabulary(train_tags)

    # Update to indexes representation.
    train_data = convert_data_to_indexes(train_data, w2i)
    train_tags = convert_tags_to_indexes(train_tags, t2i)

    # Organize the data set into context windows.
    train_contexts, train_tags = update_data_set_to_window_based(train_data, train_tags)

    # Creating a torch loader.
    train_data_set = TensorDataset(torch.LongTensor(train_contexts), torch.LongTensor(train_tags))
    train_loader = DataLoader(train_data_set, batch_size=BATCH_SIZE, shuffle=True)

    """ Handling the dev set """

    # Loading the dev set.
    dev_data, dev_tags = read_data("./" + TASK + "/dev")

    # Update to indexes representation.
    dev_data = convert_data_to_indexes(dev_data, w2i)
    dev_tags = convert_tags_to_indexes(dev_tags, t2i)

    # Organize the data set into context windows.
    dev_contexts, dev_tags = update_data_set_to_window_based(dev_data, dev_tags)

    # Creating a torch loader.
    dev_data_set = TensorDataset(torch.LongTensor(dev_contexts), torch.LongTensor(dev_tags))
    dev_loader = DataLoader(dev_data_set, batch_size=BATCH_SIZE, shuffle=True)

    """ Handling the test set """

    # Loading the test set.
    test_data, original_data = read_test_data("./" + TASK + "/test")

    # Update to indexes representation.
    test_data = convert_data_to_indexes(test_data, w2i)

    # Organize the data set into context windows.
    test_contexts = update_test_set_to_window_based(test_data)

    # Creating an instance of MlpTagger.
    model = MlpTagger(vocab_size=len(w2i), embeddings=vecs, hidden_layer_size=150, output_size=len(t2i))

    # Using Adam optimizer
    optimizer = optim.Adam(model.parameters())

    # Training the model
    epochs_Dev_Loss, epochs_Dev_Acc = train(model, optimizer, train_loader, dev_loader, EPOCHS, i2t)

    # Generate graphs describing the dev's loss and accuracy as a function on the number of epochs.
    generate_graph(output_dir, epochs_Dev_Loss, "dev_loss", "Dev Loss")
    generate_graph(output_dir, epochs_Dev_Acc, "dev_accuracy", "Dev Accuracy")

    # Calculating the predictions of the model to the test examples and writing each one to the file.
    test_predictions(test_contexts, original_data, i2t)
