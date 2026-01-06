from src.preprocess import build_features
from src.graph import build_knn_rbf_graph_small
import numpy as np
from src.graph import build_knn_rbf_graph_small, normalize_symmetric

# verify that the graph has correct dimensions (only sparse number of edges and each node cca k members)
names, X = build_features("data/anime.csv")
Xs = X.iloc[:500].values.astype(float)

W = build_knn_rbf_graph_small(Xs, k=10)
print("W shape:", W.shape)
print("Nonzeros:", W.nnz)
print("Avg degree approx:", W.nnz / W.shape[0])

# verify that the kNN graph is well-formed: (no isolated nodes, reasonable spread of connections, confirms that the neighborhood structure is stable)
deg = np.array(W.getnnz(axis=1))
print("Min degree:", deg.min())
print("Mean degree:", deg.mean())
print("Max degree:", deg.max())
print("Nodes with degree < 5:", np.sum(deg < 5))

# normalization check: test label propagation is numerically stable and high  degree nodes do not dominate propagation
S = normalize_symmetric(W)
print("S shape:", S.shape)
print("S nonzeros:", S.nnz)