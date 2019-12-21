import numpy as np

K = 5


# The function returns the cosine distance between the two vectors.
def dist(u, v):
    return np.dot(u, v) / (np.sqrt(np.dot(u, u)) * np.sqrt(np.dot(v, v)))


# The function gets a word and returns the k most similar words to it according to the cosine distance.
def most_similar(word, k):
    distances = [(w, dist(word_to_vec[word], word_to_vec[w])) for w in vocab if w != word]
    distances.sort(key=lambda x: x[1], reverse=True)
    return distances[:k]


if __name__ == '__main__':

    # Loading the pre-trained embedding vectors into a numpy array.
    vecs = np.loadtxt("wordVectors.txt")

    # Reading the words vocabulary.
    with open("vocab.txt", "r", encoding="utf-8") as file:
        vocab = file.readlines()
        vocab = [word.strip() for word in vocab]

    # Dictionary where the key is the word and the value is the correspondent embedding vector.
    word_to_vec = {word: vec for word, vec in zip(vocab, vecs)}

    # For each word in the list, print the k most similar words to it according to the cosine distance.
    for word in ["dog", "england", "john", "explode", "office"]:
        print("{0} :".format(word), end=" ")
        for i, i_similar in enumerate(most_similar(word, K), 1):
            print("{1} {2}".format(word, i_similar[0], i_similar[1]), end="\n" if i == K else ", ")
