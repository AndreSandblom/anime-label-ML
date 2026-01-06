import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from scipy.sparse import diags

# build a similarity graph where each anime is connected to its k most similar neighbours
def build_knn_rbf_graph_small(X: np.ndarray, k: int = 15, gamma: float = None) -> csr_matrix:
    """
    Build a sparse kNN + RBF affinity graph.
    Using Euclidan distance in the featre space and weighted by an RBF kernel.
    Safe for small X (<=2000 rows).
    """
    n = X.shape[0]
    # computes distance between all anime
    D = cdist(X, X, metric="euclidean")

    if gamma is None:
        vals = D[np.triu_indices(n, 1)]
        vals = vals[vals > 0]
        med = np.median(vals) if vals.size > 0 else 1.0
        gamma = 1.0 / (2 * med * med)

    rows, cols, data = [], [], []

    for i in range(n):
        # find closest anime to anime i, skip self
        nn = np.argsort(D[i])[1:k+1]  
        for j in nn:
            # converts distance -> influence
            w = np.exp(-gamma * (D[i, j] ** 2))
            rows += [i, j]
            cols += [j, i]
            data += [w, w]

    W = csr_matrix((data, (rows, cols)), shape=(n, n))
    return W

# normalize W into S and make sure "numbers" comparable (formula: S = D^{-1/2} * W * D^{-1/2})
def normalize_symmetric(W):
    """
    Normalize similarity graph so that all anime
    contribute equally during label propagation.

    So the popular or highly connected anime
    do not dominate the diffusion process.
    """
    d = np.array(W.sum(axis=1)).ravel()
    invsqrt = np.reciprocal(np.sqrt(d), where=d != 0)
    Dm12 = diags(invsqrt)
    return Dm12.dot(W).dot(Dm12)
