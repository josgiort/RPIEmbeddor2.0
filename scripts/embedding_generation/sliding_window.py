"""
sliding_window.py

Sliding window approach for embedding long protein sequences, following:
Brandes et al. (2023, Nature Genetics). Genome-wide prediction of disease
variant effects with a deep protein language model.
https://doi.org/10.1038/s41588-023-01465-0
"""

import numpy as np


DEFAULT_WINDOW_SIZE = 1022
MIN_OVERLAP = 511   


def window_weights(window_len):
    """
    Returns a numpy array of shape (window_len,) with per-position sigmoid
    weights following Brandes et al. Extended Data Fig. 5a.

    Positions are 1-based as in the paper (1 <= i <= 1022):
      - ramp up:    1 <= i < 256   -> sigmoid increasing from ~0 to 1
      - flat:     256 <= i < 766   -> weight = 1.0
      - ramp down: 766 <= i <= 1022 -> sigmoid decreasing from 1 to ~0
    """

     # scale ramp width proportionally to window size
    ramp_width  = window_len // 4          # 256 for window_len=1022
    sigmoid_center = ramp_width // 2       # 128 for window_len=1022

    i = np.arange(1, window_len + 1, dtype=np.float32)
    w = np.ones(window_len, dtype=np.float32)

    # ramp up: 1 <= i < ramp_width
    ramp_up = i < ramp_width
    w[ramp_up] = 1.0 / (1.0 + np.exp(-(i[ramp_up] - sigmoid_center) / 16))

    # flat center: ramp_width <= i < window_len - ramp_width — already 1.0, nothing to do

    # ramp down: i >= window_len - ramp_width
    ramp_down = i >= (window_len - ramp_width)
    w[ramp_down] = 1.0 / (1.0 + np.exp((i[ramp_down] - window_len + sigmoid_center) / 16))

    return w  # shape: (window_len,)


def get_window_starts(L, window_size=DEFAULT_WINDOW_SIZE, min_overlap=MIN_OVERLAP):
    """
    Tiles a sequence of length L into overlapping windows following
    Brandes et al. (2023, Nature Genetics):
      - Windows grow inward from both ends simultaneously with exactly
        511 positions of overlap between consecutive windows
      - If the overlap between the two innermost meeting windows is less
        than 511, one extra central window is inserted

    Returns a list of (start, end) tuples sorted by start position,
    using 0-based indices suitable for Python slicing.
    """
    step = window_size - min_overlap  # 511

    if L <= window_size:
        return [(0, L)]

    windows = []

    # initialise pointers at both ends
    start_left  = 0
    end_left    = window_size
    start_right = L - window_size
    end_right   = L

    # place initial windows at both ends
    windows.append((start_left,  end_left))
    windows.append((start_right, end_right))

    # grow inward from both ends simultaneously until they meet
    while end_left < start_right:
        # advance left window inward
        start_left = start_left + step
        end_left   = start_left + window_size
        windows.append((start_left, end_left))

        # advance right window inward
        start_right = start_right - step
        end_right   = start_right + window_size
        windows.append((start_right, end_right))

    # check overlap between the two innermost windows
    diff = end_left - start_right  # always >= 0 after the loop

    if diff < min_overlap:
        # insert one extra central window
        middle       = diff // 2
        middle_index = start_right + middle
        extra_start  = max(0, middle_index - step)
        extra_end    = min(extra_start + window_size, L)
        windows.append((extra_start, extra_end))

    # deduplicate and sort by start position
    windows = sorted(set(windows), key=lambda x: x[0])

    return windows


def embed_sequence_with_windows(embedding_id, sequence, embed_window_fn, window_size=DEFAULT_WINDOW_SIZE, min_overlap=MIN_OVERLAP):
    """
    Returns a numpy array of shape (L, embedding_dim) for a sequence of any
    length, using sliding window stitching with sigmoid-weighted averaging
    for sequences longer than WINDOW_SIZE.

    Args:
        embedding_id (str): identifier for the sequence, used to name windows
        sequence (str):     amino acid sequence
        embed_window_fn:    callable(seq_id, sequence) -> np.array (win_len, emb_dim)
                            should return per-residue embeddings for a single window

    Returns:
        np.array of shape (L, embedding_dim)
    """
    L = len(sequence)

    if L <= window_size:
        return embed_window_fn(embedding_id, sequence)

    windows       = get_window_starts(L,  window_size=window_size, min_overlap=min_overlap)
    accumulator   = None
    weight_sum    = None

    for idx_w, (start, end) in enumerate(windows):
        win_seq = sequence[start:end]
        win_len = end - start
        win_id  = f"{embedding_id}_w{idx_w}"

        win_emb = embed_window_fn(win_id, win_seq)
        # win_emb shape: (win_len, embedding_dim)

        if accumulator is None:
            embedding_dim = win_emb.shape[1]
            accumulator   = np.zeros((L, embedding_dim), dtype=np.float32)
            weight_sum    = np.zeros((L, 1),             dtype=np.float32)

        # per-position sigmoid weights, broadcast across embedding dimensions
        w = window_weights(win_len)   # shape: (win_len,)
        w = w[:, np.newaxis]          # shape: (win_len, 1)

        accumulator[start:end] += win_emb * w
        weight_sum[start:end]  += w

    # normalised weighted average
    return accumulator / weight_sum   # shape: (L, embedding_dim)