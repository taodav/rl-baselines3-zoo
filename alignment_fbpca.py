import numpy as np
import fbpca
from typing import Optional, Tuple

def _iter_cols(A: np.ndarray, block_cols: int):
    """Yield (s, e, A[:, s:e]) over columns in blocks."""
    D, N = A.shape
    for s in range(0, N, block_cols):
        e = min(N, s + block_cols)
        yield s, e, A[:, s:e]

def _center_mean_over_cols(A: np.ndarray, block_cols: int, dtype=np.float32):
    """Return per-row mean over columns, as (D,1) for X or (C,1) for Y."""
    D, N = A.shape
    m = np.zeros((D,), dtype=np.float64)
    for _, _, Ab in _iter_cols(A, block_cols):
        m += Ab.sum(axis=1, dtype=np.float64)
    m /= float(N)
    return m.astype(dtype)[:, None]  # (D,1)

def _matmul_centered_X_times(B: np.ndarray, X: np.ndarray, muX: np.ndarray, block_cols: int):
    """
    Compute (X - muX @ 1^T) @ B, in blocks over columns of X.
    X: (D,N), muX: (D,1), B: (N,k)  ->  returns (D,k)
    """
    D, N = X.shape
    out = np.zeros((D, B.shape[1]), dtype=np.float32)
    col0 = 0
    for s, e, Xb in _iter_cols(X, block_cols):
        xb = Xb.astype(np.float32) - muX  # (D, b)
        Bb = B[s:e, :]                     # (b, k)
        out += xb @ Bb
        col0 += e - s
    return out

def _matmul_centered_XT_times(Q: np.ndarray, X: np.ndarray, muX: np.ndarray, block_cols: int):
    """
    Compute (X - muX @ 1^T)^T @ Q, in blocks over columns of X.
    X: (D,N), muX:(D,1), Q:(D,q)  -> returns (N,q)
    """
    D, N = X.shape
    out = np.zeros((N, Q.shape[1]), dtype=np.float32)
    row = 0
    for s, e, Xb in _iter_cols(X, block_cols):
        xb = Xb.astype(np.float32) - muX  # (D, b)
        out[s:e, :] = xb.T @ Q            # (b, q)
        row += e - s
    return out

def _YYt_centered(Y: np.ndarray, muY: np.ndarray, block_cols: int):
    """
    Compute Yc Yc^T = sum_b (Yb - muY)(Yb - muY)^T  without materializing Yc.
    Returns (C,C).
    """
    C, N = Y.shape
    YYt = np.zeros((C, C), dtype=np.float32)
    for _, _, Yb in _iter_cols(Y, block_cols):
        ybc = Yb.astype(np.float32) - muY
        YYt += ybc @ ybc.T
    return YYt

def alignment_sweep_fbpca(
    X: np.ndarray,         # (D, N)  columns = samples
    Y: np.ndarray,         # (C, N)
    k: int,                # number of leading components to approximate
    major: str = "C",      # "C": sample-space (V); "R": feature-space (U)
    center: bool = True,
    block_cols: int = 8192,
    n_iter: int = 2,       # power iterations for fbpca
    oversample: int = 10,  # oversampling for fbpca
) -> np.ndarray:
    """
    Alignment sweep using randomized SVD (fbpca), memory-lean & chunked.

    Returns: alignment curve (length = k).
    """
    assert X.ndim == 2 and Y.ndim == 2
    D, N = X.shape
    C, Ny = Y.shape
    assert N == Ny, "X and Y must have the same number of samples (columns)."
    assert k <= min(D, N), "k must be <= min(D, N)."

    # ---- means (no big copies) ----
    muX = _center_mean_over_cols(X, block_cols) if center else np.zeros((D, 1), dtype=np.float32)
    muY = _center_mean_over_cols(Y, block_cols) if center else np.zeros((C, 1), dtype=np.float32)

    # ---- fbpca randomized SVD on centered X, without materializing Xc ----
    # fbpca needs A @ Omega and A.T @ Q. We implement those by lambdas that
    # stream over columns of X and subtract muX on the fly.

    def A_dot(B: np.ndarray) -> np.ndarray:
        # B: (N, q) -> (D, q)
        return _matmul_centered_X_times(B, X, muX, block_cols)

    def AT_dot(Q: np.ndarray) -> np.ndarray:
        # Q: (D, q) -> (N, q)
        return _matmul_centered_XT_times(Q, X, muX, block_cols)

    # Call fbpca.pca with function handles (raw=True keeps thin factors)
    U, s, Vt = fbpca.pca(
        A=(A_dot, AT_dot, (D, N)),
        k=k,
        raw=True,
        n_iter=n_iter,
        l=k + oversample
    )
    V = Vt.T.astype(np.float32)        # (N, k)
    U = U.astype(np.float32)           # (D, k)
    s = s.astype(np.float32)           # (k,)

    # ---- Denominator: ||Yc Yc^T||_F^2 (small CxC) ----
    YYt = _YYt_centered(Y, muY, block_cols)     # (C, C)
    denom = float(np.linalg.norm(YYt, ord="fro")**2) + 1e-12

    # ---- Numerator (two streamed passes over Y), block all k at once ----
    # First pass: T = Yc @ V  (C x k)
    T = np.zeros((C, k), dtype=np.float32)
    for s0, e0, Yb in _iter_cols(Y, block_cols):
        ybc = Yb.astype(np.float32) - muY            # (C, b)
        Vb  = V[s0:e0, :]                             # (b, k)
        T  += ybc @ Vb                                # (C, k)

    # Second pass: sum over blocks of || (Yc^T T)_block ||^2 per column
    numer = np.zeros((k,), dtype=np.float64)
    for _, _, Yb in _iter_cols(Y, block_cols):
        ybc  = Yb.astype(np.float32) - muY           # (C, b)
        dots = ybc.T @ T                              # (b, k)
        numer += np.sum(dots.astype(np.float64)**2, axis=0)

    curve = np.cumsum(numer) / denom
    return curve.astype(np.float64)




if __name__ == "__main__":
    dataset_fname = "/home/taodav/Documents/rl-baselines3-zoo/buffers/BreakoutNoFrameskip-v4-ppo-100000_compressed/buffer.npz"

    env_name = '-'.join(dataset_fname.split('/')[-2].split('-')[:2])

    buffer = np.load(dataset_fname)
    states = buffer['observations']
    # mean_pooled_states = downsample_mean(states)

    # compressed_dataset_fname = dataset_fname + "_compressed"
    # buffer['observations'] = mean_pooled_states
    # np.savez_compressed(compressed_dataset_fname, **buffer)

    flat_states = states.reshape(states.shape[0], -1).T
    returns = buffer['returns'][:, None].T

    return_alignments = alignment_sweep_dask(flat_states, returns, 
                                             k=flat_states.shape[0],
                                             randomized=True,
                                             chunks=(flat_states.shape[0], 500))

    fig, ax = plt.subplots()

    ax.plot(return_alignments)
    ax.set_ylim([0, 1])
    ax.set_title(f"{ckpt_args['env']} alignment over k")
    plt.show()
    print()