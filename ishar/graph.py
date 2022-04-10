import numpy as np

class Graph:
    v: int # Number of vertices

    def __init__(self, vertices):
        self.nodes = vertices
        self.v = len(self.nodes)
        self.graph = np.zeros((self.v, self.v), dtype='float')

    def add_edge(self, w1, w2):
        i, j = self.nodes[w1], self.nodes[w2]
        self.graph[i][j] = 1

    def add_edges(self, edges, undirected=False):
        for edge in edges:
            self.add_edge(*edge)
        
        if undirected:
            self.__symmetrize()
        return self.graph
    
    def normalize(self):
        norm = np.sum(self.graph, axis=0)
        self.graph =  np.divide(self.graph, norm, where=norm!=0)
        return self.graph
    
    def __symmetrize(self):
        self.graph += self.graph.T - np.diag(self.graph.diagonal())
        return self.graph

    def __repr__(self):
        result = ""
        for n, r in zip(self.nodes, self.graph):
            result += str(n) + str(r) + "\n"
        return result
