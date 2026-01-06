import numpy as np
from src.preprocess import build_features
from src.graph import build_knn_rbf_graph_small, build_knn_rbf_graph_chunked, normalize_symmetric

# check that W_chunk has similar stats to W_small, min degree close to k, no nodes with degree < 5
def degree_stats(M):
    deg = np.array(M.getnnz(axis=1))
    return deg.min(), deg.mean(), deg.max(), (deg < 5).sum()

names, X = build_features("data/anime.csv")
# test the chunked graph on 500 rows
# Xs = X.iloc[:500].values.astype(float)
# test the chunked graph on 2000 rows and pray for no crash
Xs = X.iloc[:2000].values.astype(float)
k = 15
# k = 10

print("=== SMALL GRAPH (FULL DISTANCE) ===")
W_small = build_knn_rbf_graph_small(Xs, k=k)
print("W_small nnz:", W_small.nnz)
print("deg(min/mean/max, <5):", degree_stats(W_small))

print("\n=== CHUNKED GRAPH (SAFE) ===")
W_chunk = build_knn_rbf_graph_chunked(Xs, k=k, chunk_size=200)
print("W_chunk nnz:", W_chunk.nnz)
print("deg(min/mean/max, <5):", degree_stats(W_chunk))

print("\n=== NORMALIZATION CHECK ===")
S_small = normalize_symmetric(W_small)
S_chunk = normalize_symmetric(W_chunk)
print("S_small nnz:", S_small.nnz)
print("S_chunk nnz:", S_chunk.nnz)

# compare rough similarity of degree distribution (should be close)
deg_small = np.array(W_small.getnnz(axis=1))
deg_chunk = np.array(W_chunk.getnnz(axis=1))
print("\nDegree diff (mean abs):", np.mean(np.abs(deg_small - deg_chunk)))

# verify kNN selection is correct (expected 0 or low, high = bug)
print("\n=== CHECK: each row has at least k neighbors ===")
deg = np.array(W_chunk.getnnz(axis=1))
print("Rows below k:", np.sum(deg < k))




