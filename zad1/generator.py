import bs4
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from collections import Counter
import scipy
import numpy as np
import pickle
import sys
import grequests


def main():
    output = "data.dat"
    n = 100
    if len(sys.argv) > 2:
        output = sys.argv[2]
    if len(sys.argv) > 1:
        n = int(sys.argv[1])

    i = 0
    urls_left = set(["https://en.wikipedia.org/"])
    urls_visited = set()
    urls_indexes = dict()
    words = set()
    words_indexes = dict()
    freq_matrix = []
    while len(urls_left) > 0 and i < n:
        urls = []
        for _ in range(min(len(urls_left), 50, n - i)):
            url = urls_left.pop()
            urls.append(url)
            urls_visited.add(url)
        rs = (grequests.get(u) for u in urls)
        responses = grequests.map(rs)
        for response in responses:
            if response is None:
                print("REQUEST ERROR")
                continue

            url = response.url
            soup = bs4.BeautifulSoup(response.text, "html.parser")

            new_urls = set(get_urls(soup))
            urls_left = urls_left.union(new_urls.difference(urls_visited))
            urls_indexes[i] = url

            page_words = get_words(soup)
            new_words = set(page_words).difference(words)
            for new_word in new_words:
                words_indexes[new_word] = len(words)
                words.add(new_word)

            if i > 0 and len(new_words) > 0:
                new_rows = scipy.sparse.lil_matrix(np.zeros((len(new_words), n)))
                freq_matrix = scipy.sparse.vstack([freq_matrix, new_rows], "lil")
            elif i == 0:
                freq_matrix = scipy.sparse.lil_matrix((len(new_words), n))

            freqs = Counter(page_words)
            for word, freq in freqs.items():
                freq_matrix[words_indexes[word], i] = freq

            i += 1
            print(str(i) + " " + url)

    data = {'words_indexes': words_indexes, 'urls_indexes': urls_indexes}

    data['freq_matrix'] = freq_matrix.tocsc()
    normalize_csc(data['freq_matrix'])
    with open("norm_" + output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Results saved to file: norm_" + output + "\n")

    idf(freq_matrix)
    data['freq_matrix'] = freq_matrix.tocsc()
    normalize_csc(data['freq_matrix'])
    with open("idf_" + output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Results saved to file: idf_" + output + "\n")

    m = svd(freq_matrix, min(100, n - 1))
    normalize_csc(m)
    data['freq_matrix'] = m
    with open("svd_" + output, "wb") as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    print("Results saved to file: svd_" + output + "\n")


def get_words(soup):
    p_list = soup.find_all('p')
    text = [list(p.strings) for p in p_list]
    text = ' '.join([t for l in text for t in l])
    tokens = nltk.word_tokenize(text)
    stop = stopwords.words('english')
    stemmer = SnowballStemmer("english")
    tokens = [stemmer.stem(x) for x in tokens if len(x) > 2 and x not in stop]
    return tokens


def get_urls(soup):
    urls = [link.get('href') for link in soup.find_all('a')
            if link.get('href') is not None]
    filtered = []
    for url in urls:
        if url.find('File:') != -1:
            continue
        if url.find('?') != -1:
            url = url[0:url.find('?')]
        if url[0] != '/' and url.find('en.wikipedia.org') != -1:
            filtered.append(url)
        elif url[0] == '/' and url[1] != '/':
            filtered.append('https://en.wikipedia.org' + url)
        elif url.find('en.wikipedia.org') != -1:
            filtered.append(url.strip('/'))

    return filtered


def normalize_csc(matrix):
    print("Performing normalization...")
    words, pages = matrix.shape
    for i in range(pages):
        norm = np.linalg.norm(matrix.data[matrix.indptr[i]:matrix.indptr[i+1]])
        if norm == 0:
            continue
        for j in range(matrix.indptr[i], matrix.indptr[i+1]):
            matrix.data[j] /= norm
        if (i+1) % 100 == 0:
            print(str(i + 1) + " normalized")
    print("Normalization successful.")


def idf(matrix):
    words, pages = matrix.shape
    for i in range(words):
        nw = len(matrix.rows[i])
        matrix[i, :] *= np.log10(pages/nw)
        if (i+1) % 500 == 0:
            print(str(i + 1) + " idf multiplied")
    print("Multiplied by idf.")


def svd(matrix, k):
    print("Performing svd...")
    u, s, v = scipy.sparse.linalg.svds(matrix, k=k)
    m = scipy.sparse.csc_matrix(u.dot(np.diag(s)).dot(v))
    print("Svd successful.")
    return m


if __name__ == "__main__":
    main()
