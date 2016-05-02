import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from collections import Counter
import pickle
import sys
import heapq


def main():
    input = "data.dat"

    if len(sys.argv) > 1:
        input = sys.argv[1]
    with open(input, 'rb') as infile:
        data = pickle.load(infile)
    words_indexes = data['words_indexes']
    freq_matrix = data['freq_matrix']
    urls_indexes = data['urls_indexes']

    print("Enter keywords:")
    query = sys.stdin.readline()
    words = get_words(query)
    freqs = Counter(words)
    query_col = np.zeros((len(words_indexes), 1))
    for word, freq in freqs.items():
        if word in words_indexes:
            query_col[words_indexes[word], 0] = freq

    query_col_t = np.transpose(query_col) / np.linalg.norm(query_col)
    n = 10
    results = dict()
    for page_index, page in urls_indexes.items():
        d = np.matrix(freq_matrix[:, page_index].toarray())
        results[page] = (query_col_t * d)[0, 0]

    query_urls = heapq.nlargest(n, results.keys(), key=lambda x: results[x])
    for url in query_urls:
        print(url)


def get_words(query):
    tokens = nltk.word_tokenize(query)
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(x) for x in tokens if len(x) > 2]
    return tokens


if __name__ == "__main__":
    main()
