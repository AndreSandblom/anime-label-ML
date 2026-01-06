import numpy as np

def spread_labels(S, Y, alpha=0.8, max_iter=50, tol=1e-6):
    """
    Manual label spreading algorithm.

    Parameters:
        S : (n x n) normalized similarity graph
        Y : (n x c) initial label matrix
        alpha : propagation strength (0.8 typical)
        max_iter : maximum iterations
        tol : convergence tolerance

    Returns:
        F : (n x c) propagated soft label matrix
    """
    F = Y.astype(float).copy()

    for it in range(max_iter):
        F_new = alpha * (S @ F) + (1 - alpha) * Y
        diff = np.max(np.abs(F_new - F))
        F = F_new
        if diff < tol:
            print(f"Converged at iteration {it}, diff={diff:.2e}")
            break

    # Normalize rows to sum to 1 (probabilities)
    row_sum = F.sum(axis=1, keepdims=True)
    # Unreachable nodes: never received any label signal => row_sum == 0
    unreachable = (row_sum.ravel() == 0)

    # Avoid division by zero
    row_sum[row_sum == 0] = 1.0
    F = F / row_sum

    # If unreachable, assign uniform probability (model says "unknown")
    if np.any(unreachable):
        F[unreachable] = 0.5

    return F
