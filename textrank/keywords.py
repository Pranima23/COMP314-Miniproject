from collections import OrderedDict

import numpy as np
import spacy
from spacy.lang.en.stop_words import STOP_WORDS

from graph import Graph

nlp = spacy.load("en_core_web_sm")

def _set_stopwords(stop_words):
    """ Set additional stop words """
    global STOP_WORDS
    STOP_WORDS = STOP_WORDS.union(set(stop_words))

def _get_sentences(text, valid_pos):
    """ Returns list of validated sentence """
    doc = nlp(text)
    sents = doc.sents
    sentences = []
    for sent in sents:
        words = []
        for w in sent:
            if w.pos_ in valid_pos and w not in STOP_WORDS and w.text.isalpha():
                words.append(w.text.lower())
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

def _pagerank(graph, d, steps, convergence):
    """ Displays and returns ranks for keywords """
    weights = _calculate_pr_weights(graph, d, steps, convergence)

    ranks = {w: weights[i] for w, i in graph.nodes.items()}
    ranks =  OrderedDict(sorted(ranks.items(), key=lambda m: m[1], reverse=True))

    print("Ranks: Word - Weight")
    for i, (k, v) in enumerate(ranks.items()):
        print(f"{i+1}. {k} - {v}")
    
    return ranks

def _combine_adjacent_keywords(ranks, text, total):
    """ Combine adjancent keywords in a sentence """
    keywords = []
    words = []
    kranks = ranks.keys()
    ranks = dict(ranks)
    sents = [[word.strip().lower() for word in sent.split(" ")] for sent in text.split(".")]
    for sent in sents:
        word = []
        l = len(sent)
        for i in range(l):
            if sent[i] in kranks and sent[i] not in words:
                word.append(sent[i])
                words.append(sent[i])
                if not i + 1 == l:
                    for j in range(i+1, l):
                        if sent[j] in kranks:
                            word.append(sent[j])
                            words.append(sent[j])
                        else:
                            break
                pw = sum([ranks[w] for w in word])    
                keywords.append((" ".join(word), pw))
                word = []
    keywords = sorted(keywords, key=lambda x: x[1], reverse=True)
    return keywords[:total]

def extract_keywords(text, stop_words=list(), valid_pos= ("NOUN", "PROPN", "ADJ",), window=4, d=0.85, steps=10, convergence=1e-6, total=0):
    """ Keyword Extraction using TextRank algorithm """

    ### Pre processing ###
    # Add stop words if any
    _set_stopwords(stop_words)

    # Segment sentences through valid pos
    sentences = _get_sentences(text, valid_pos)

    # Get dict of tokens -> lemma (Vertices)
    tokens = _get_tokens(sentences)

    ### Graph Formation ###
    # Build token pairs between windows in sentences (Edges)
    token_pairs = _get_token_pairs(window, sentences)

    # Build Graph: tokens = Vertices && token_pairs = Edges
    graph = _build_graph_matrix(tokens, token_pairs)

    ### Ranking Algorithm ###
    # Implement Pagerank/Textrank algorithm to rank words
    ranks = _pagerank(graph, d, steps, convergence,)

    ### Post processing ###
    # Combine adjacent keywords
    keywords = _combine_adjacent_keywords(ranks, text, total or graph.v // 3)

    return keywords

if __name__ == "__main__":
    extract_keywords("Ram is a good boy but Hari is a bad boy.")