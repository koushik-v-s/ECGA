import numpy as np

def degree_centrality_baseline(graph, k: int):
    degree_dict = dict(graph.out_degree())
    return sorted(degree_dict, key=degree_dict.get, reverse=True)[:k]

def random_selection_baseline(graph, k: int):
    return list(np.random.choice(list(graph.nodes()), k, replace=False))
