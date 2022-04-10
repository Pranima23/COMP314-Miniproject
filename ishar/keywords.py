import numpy as np
from collections import OrderedDict
from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
from .graph import Graph

stopwords = set(stopwords.words('english'))

def _set_stopwords(stop_words):
    """ Set additional stop words """
    global stopwords
    stopwords = stopwords.union(set(stop_words))

def _get_sentences(text, valid_pos):
    """ Returns list of validated sentence """
    sents = sent_tokenize(text)
    sentences = []
    for sent in sents:
        words = []
        for w, pos in pos_tag(word_tokenize(sent)):
            if pos in valid_pos and w not in stopwords and w.isalpha():
                words.append(w)
        sentences.append(words)
    return sentences

def _get_tokens(sentences):
    """ Returns OrderedDict of individual words/tokens """
    words = OrderedDict()
    i = 0
    for sentence in sentences:
        for word in sentence:
            if word not in words:
                words[word] = i
                i += 1
    return words

def _get_token_pairs(window, sentences):
    """ Returns pairs of tokens, in span of window size in sentence"""
    token_pairs = []
    for sentence in sentences:
        for i in range(len(sentence)):
            for j in range(i+1, i+window):
                if j >= len(sentence):
                    break
                pair = sentence[i], sentence[j]
                if pair not in token_pairs:
                    token_pairs.append(pair)
    return token_pairs

def _build_graph_matrix(tokens, token_pairs):
    """ Build graph adjacency matrix, return graph object """
    graph = Graph(tokens)
    graph.add_edges(token_pairs, undirected=True)

    # Normalize the elements
    g = graph.normalize()

    return graph

def _calculate_pr_weights(graph, d, steps, convergence):
    """ Pagerank Algorithm to calculate weight of each token """
    pr = np.array([1] * graph.v)
    prev_pr = 0

    for _ in range(steps):
        pr = (1-d) + d * np.dot(graph.graph, pr)
        if abs(prev_pr - sum(pr)) < convergence:
            break
        else:
            prev_pr = sum(pr)
    return pr

def _pagerank(graph, d, steps, convergence, total):
    """ Displays and returns ranks for keywords """
    weights = _calculate_pr_weights(graph, d, steps, convergence)
    
    ranks = {w: weights[i] for w, i in graph.nodes.items()}
    ranks =  OrderedDict(sorted(ranks.items(), key=lambda m: m[1], reverse=True)[:total])

    print("Ranks: Word - Weight")
    for i, (k, v) in enumerate(ranks.items()):
        print(f"{i+1}. {k} - {v}")
    
    return ranks

def extract_keywords(text, stop_words=list(), valid_pos= ("NN", "NNP", "JJ",), window=4, d=0.85, steps=10, convergence=1e-5, total=10):
    """ Keyword Extraction using TextRank algorithm """

    # Add stop words if any
    _set_stopwords(stop_words)

    # Segment sentences through valid pos
    sentences = _get_sentences(text, valid_pos)

    # Get dict of tokens -> lemma (Vertices)
    tokens = _get_tokens(sentences)

    # Build token pairs between windows in sentences (Edges)
    token_pairs = _get_token_pairs(window, sentences)

    # Build Graph: tokens = Vertices && token_pairs = Edges
    graph = _build_graph_matrix(tokens, token_pairs)

    # Implement Pagerank/Textrank algorithm to rank words
    ranks = _pagerank(graph, d, steps, convergence, total)
    return ranks

if __name__ == "__main__":
    extract_keywords("""Ram is a very good boy unlike Hari who is a bad boy.""")