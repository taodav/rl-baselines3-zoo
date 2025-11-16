import argparse
from typing import Optional, Union
from pathlib import Path

from dask.diagnostics import ProgressBar
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
from tqdm import trange


def downsample_mean(
    X: Union[np.ndarray, np.memmap],
    chunk_size: int = 512,
    out: Optional[Union[np.ndarray, np.memmap]] = None,
    dtype=np.float32,
):
    """
    X: (B,84,84,4) array-like (np.ndarray or np.memmap). Columns are channels last.
    chunk_size: number of items per chunk along batch dimension.
    out: optional preallocated/memmap array of shape (B,42,42,4). If None, allocates.
    dtype: dtype for computation/output.

    Returns: out (np.ndarray or np.memmap) with shape (B,42,42,4)
    """
    assert X.ndim == 4 and X.shape[1:4] == (84, 84, 4), "Expected (B,84,84,4)"
    B = X.shape[0]

    if out is None:
        out = np.empty((B, 42, 42, 4), dtype=dtype)

    print("Downsampling Images")
    for start in trange(0, B, chunk_size):
        stop = min(start + chunk_size, B)
        xb = np.asarray(X[start:stop], dtype=dtype)  # (b,84,84,4)
        # 2x2 area/mean pooling without copies
        xb = xb.reshape(stop-start, 42, 2, 42, 2, 4).mean(axis=(2, 4))
        out[start:stop] = xb  # (b,42,42,4)

    return out


def alignment_sweep_dask(
    X, Y, major="C", center=True, k=None, compute=True, randomized=True,
    chunks=None,
):
    """
    Alignment sweep using Dask with memory-efficient centering and (truncated) SVD.

    Parameters
    ----------
    X : (D, N) array-like
        Dask array preferred. Columns are samples.
    Y : (C, N) array-like
        Dask array preferred. Rows are targets (C=1 for scalar regression).
    major : {"C","R"}
        "C": use top-k in sample space (right singular vectors V).
        "R": use top-k in feature space (left singular vectors U).
    center : bool
        Center X and Y across samples (subtract column means per row).
    k : int or None
        Number of leading components. If None, tries full SVD (expensive).
    compute : bool
        If True, returns a NumPy array. If False, returns a Dask array (lazy).
    randomized : bool
        If True and k is not None, uses randomized/truncated SVD (svd_compressed).
    chunks : tuple or None
        Optional chunks for X/Y (e.g., (D, 10000)). If None, leaves as-is.

    Returns
    -------
    curve : 1D array (length = k or rank)
        Cumulative alignment curve.
    """
    X = da.asarray(X)
    Y = da.asarray(Y)

    # Ensure both share the same column chunking (axis=1) for efficient matmuls
    if chunks is not None:
        # Rechunk only along columns; keep full rows to avoid duplicating along axis=0
        X = X.rechunk({0: X.shape[0], 1: chunks[1] if len(chunks) > 1 else chunks})
        Y = Y.rechunk({0: Y.shape[0], 1: X.chunksize[1]})
    else:
        # Best-effort: align Y's column chunks to X's
        Y = Y.rechunk({1: X.chunksize[1]})

    D, N = X.shape
    C, Ny = Y.shape
    assert N == Ny, "X and Y must have the same number of samples (columns)."

    # ---- Center (memory-efficient) ----
    if center:
        # Per-row mean over columns
        mu_X = X.mean(axis=1, keepdims=True)   # (D,1)
        mu_Y = Y.mean(axis=1, keepdims=True)   # (C,1)
        Xc = X - mu_X
        Yc = Y - mu_Y
    else:
        Xc, Yc = X, Y

    # ---- SVD / PCs ----
    # We need either V (N×r) for major="C", or U (D×r) for major="R".
    if k is None:
        # Full SVD (expensive for large problems). Dask will do a blocked algorithm.
        U, S, Vt = da.linalg.svd(Xc)
        V = Vt.T
    else:
        if randomized:
            # Randomized truncated SVD: fast and memory-friendly for large matrices
            U, S, Vt = da.linalg.svd_compressed(Xc, k=k, n_oversamples=8)
            V = Vt.T
        else:
            # Deterministic tall-skinny SVD route (only efficient if D >> N or vice versa)
            U, S, Vt = da.linalg.svd(Xc)
            V = Vt.T
            if k is not None:
                U, S, V = U[:, :k], S[:k], V[:, :k]

    r = V.shape[1] if major == "C" else U.shape[1]

    # ---- Denominator: ||Y Y^T||_F^2 (small C×C) ----
    # Frobenius norm invariant under transpose, so we avoid N×N.
    YYt = Yc @ Yc.T          # (C, C)
    denom = da.linalg.norm(YYt, ord="fro") ** 2  # scalar (dask)

    # ---- Numerator per component, no N×N allocations ----
    if major == "C":
        # M = Y^T @ (Y @ V)  -> (N, r), columns correspond to components
        # Compute in two multiplies to avoid Y^T Y.
        YV = Yc @ V               # (C, r)
        M  = Yc.T @ YV            # (N, r)
        # Column-wise squared norms
        numer = da.linalg.norm(M, axis=0) ** 2  # (r,)
    else:
        # "R" branch uses U instead of V: M = (Y Y^T) @ U, then col-norms
        M = (Yc @ Yc.T) @ U       # (C, r)
        numer = da.linalg.norm(M, axis=0) ** 2  # (r,)

    curve = da.cumsum(numer) / denom            # (r,)

    with ProgressBar():
        result = curve.compute() if compute else curve

    return result



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute alignment of returns and states.")
    parser.add_argument("--dataset-fname", type=str)
    parser.add_argument("--x_include_actions", action="store_true", default=False)
    parser.add_argument("--target", type=str, default="returns", choices=["returns", "next_obs", "policy_gradient"])
    
    args = parser.parse_args()

    dataset_fname = args.dataset_fname
    dataset_path = Path(dataset_fname)

    env_name = '-'.join(dataset_fname.split('/')[-2].split('-')[:2])

    buffer = dict(np.load(dataset_fname))
    states = buffer['observations']

    # mean_pooled_states = downsample_mean(states)
    # dataset_path = Path(dataset_fname)
    # parent_dir = dataset_path.parent
    # new_parent_dir = parent_dir.parent / (parent_dir.name + "_compressed")
    # new_parent_dir.mkdir(parents=True, exist_ok=True)
    # new_dataset_path = new_parent_dir / dataset_path.name
    # buffer['observations'] = mean_pooled_states
    # np.savez_compressed(new_dataset_path, **buffer)

    flat_states = states.reshape(states.shape[0], -1).T  # D x N

    if args.x_include_actions:
        actions = buffer['actions'].T  # A x N
        flat_states = np.concatenate([flat_states, actions], axis=1)

    
    if args.target == "returns":
        targets = buffer['returns'][:, None].T  # 1 x N
    elif args.target == "next_obs":
        targets = buffer['next_observations'].reshape(buffer['next_observations'].shape[0], -1).T  # D x N
    elif args.target == "policy_gradient":
        returns = buffer['returns'][None, :]
        log_probs = buffer['log_probs'][None, :]
        pg_targets = returns * log_probs

    k = flat_states.shape[0]
    return_alignments = alignment_sweep_dask(flat_states, returns, 
                                             k=k,
                                             randomized=True,
                                             chunks=(flat_states.shape[0], 500))

    results = {
        'return_alignments': return_alignments,
        'k': k
    }
    alignments_fpath = dataset_path.parent / "alignments.npy"

    np.save(alignments_fpath, results)
    print(f"Saved alignment results to {alignments_fpath}")
    # fig, ax = plt.subplots()

    # ax.plot(return_alignments)
    # ax.set_ylim([0, 1])
    # ax.set_title(f"{ckpt_args['env']} alignment over k")
    # plt.show()
    # print()