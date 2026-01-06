import os
import numpy as np
import pandas as pd

from src.preprocess import build_features
from src.graph import build_knn_rbf_graph_chunked, normalize_symmetric
from src.label_spreading import spread_labels


def run_full_pipeline(
    anime_csv: str,
    seed_labels_csv: str,
    outdir: str = "outputs",
    k: int = 15,
    alpha: float = 0.8,
    chunk_size: int = 200,
    limit_rows: int | None = None,
):
    """
    End-to-end pipeline:
      1) preprocess features
      2) build chunked kNN-RBF graph W
      3) normalize to S
      4) create label matrix Y from seed labels
      5) run manual label spreading
      6) save predictions CSV
    """
    print("1) Loading & preprocessing features...")
    names, Xdf = build_features(anime_csv)

    if limit_rows is not None:
        print(f"   Using subset limit_rows={limit_rows}")
        names = names.iloc[:limit_rows].reset_index(drop=True)
        Xdf = Xdf.iloc[:limit_rows].reset_index(drop=True)

    X = Xdf.values.astype(float)
    n = len(names)
    print("   Feature matrix:", X.shape)

    print("2) Loading seed labels...")
    labs = pd.read_csv(seed_labels_csv)
    labs["key"] = labs["name"].astype(str).str.lower().str.strip()
    key = names.astype(str).str.lower().str.strip()

    # y: -1 means unlabeled, 0 sad, 1 happy
    y = np.full(n, -1, dtype=int)
    mapping = dict(zip(labs["key"], labs["happy_ending"]))

    for i, nk in enumerate(key):
        if nk in mapping:
            y[i] = int(mapping[nk])

    labeled_mask = (y != -1)
    print("   Seed labels matched:", labeled_mask.sum(), "/", len(labs))
    assert labeled_mask.sum() >= 45, "Too few seed labels matched — check name alignment!"

    missing = labs[~labs["key"].isin(set(key))]
    if len(missing) > 0:
        print("Missing seed titles (not found in anime.csv subset):")
        print(missing["name"].tolist())


    print("3) Building chunked kNN-RBF graph W...")
    W = build_knn_rbf_graph_chunked(X, k=k, chunk_size=chunk_size)

    deg = np.array(W.getnnz(axis=1))
    print("   Degree stats (min/mean/max):", deg.min(), deg.mean(), deg.max())
    assert deg.min() >= k, "Some nodes have fewer than k neighbors — check kNN building."

    print("4) Normalizing graph to S...")
    S = normalize_symmetric(W)

    print("5) Building initial label matrix Y...")
    # 2 classes: [sad, happy]
    Y = np.zeros((n, 2), dtype=float)
    Y[labeled_mask, y[labeled_mask]] = 1.0

    print("6) Running manual label spreading...")
    F = spread_labels(S, Y, alpha=alpha, max_iter=60)

    row_sums = F.sum(axis=1)
    unreachable_count = int(np.sum(row_sums == 0))
    print("   Prob sums (min/mean/max):", row_sums.min(), row_sums.mean(), row_sums.max())
    print("   Unreachable nodes (no path to any seed label):", unreachable_count)

    pred = F.argmax(axis=1)     # 0/1
    conf = F.max(axis=1)

    os.makedirs(outdir, exist_ok=True)
    out = pd.DataFrame({"name": names, "pred": pred, "confidence": conf})
    out_path = os.path.join(outdir, "predictions.csv")
    out.to_csv(out_path, index=False)

    print(f"Saved {out_path}")
    return out
