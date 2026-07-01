import torch


def monotonic_alignment_search(scores, token_mask=None, frame_mask=None):
    """
    Monotonic Alignment Search (MAS).

    Finds the optimal monotonic alignment that maximizes the sum of scores,
    where each frame aligns to exactly one token and the alignment is non-decreasing.

    Args:
        scores: (B, T_spec, T_text) — log-scale alignment scores (higher = better)
        token_mask: (B, T_text) or None — True for padding tokens
        frame_mask: (B, T_spec) or None — True for padding frames (these are excluded from DP)

    Returns:
        alignment: (B, T_spec) — token index for each frame (dtype=torch.long)
    """
    B, T_spec, T_text = scores.shape
    device = scores.device
    alignment = torch.zeros(B, T_spec, dtype=torch.long, device=device)

    for b in range(B):
        s = scores[b]
        if token_mask is not None:
            s = s.masked_fill(token_mask[b].unsqueeze(0), float("-inf"))

        # Determine number of valid frames
        if frame_mask is not None:
            n_valid = (~frame_mask[b]).sum().item()
        else:
            n_valid = T_spec

        if n_valid < 1 or T_text < 1:
            continue

        # Only run DP on valid frames
        s_valid = s[:n_valid]
        nv = n_valid

        # ── Forward DP (Viterbi variant: max over paths) ──
        Q = torch.full((nv, T_text), float("-inf"), device=device)
        Q[0, 0] = s_valid[0, 0]

        for i in range(1, nv):
            Q[i, 0] = s_valid[i, 0] + Q[i - 1, 0]
            for j in range(1, min(i + 1, T_text)):
                stay = Q[i - 1, j]
                advance = Q[i - 1, j - 1]
                Q[i, j] = s_valid[i, j] + (stay if stay > advance else advance)

        # ── Backtrack ──
        last_token = (~token_mask[b]).sum().item() - 1 if token_mask is not None else T_text - 1
        last_token = max(last_token, 0)

        align = torch.zeros(nv, dtype=torch.long, device=device)
        align[-1] = last_token

        for i in range(nv - 2, -1, -1):
            j_next = align[i + 1]
            stay = Q[i, j_next]
            advance = Q[i, max(j_next - 1, 0)]
            if advance > stay and j_next > 0:
                align[i] = j_next - 1
            else:
                align[i] = j_next

        # Place alignment for valid frames
        if frame_mask is not None:
            valid_idx = ~frame_mask[b]
            alignment[b, valid_idx] = align
            # For padding frames, set to the last valid token (won't be used)
            alignment[b, frame_mask[b]] = last_token
        else:
            alignment[b] = align

    return alignment


def extract_durations(alignment, token_mask=None, min_duration=1):
    """
    Extract per-token frame counts from a MAS alignment.

    Args:
        alignment: (B, T_spec) — token index per frame
        token_mask: (B, T_text) or None — True for padding tokens
        min_duration: minimum duration for non-padding tokens

    Returns:
        durations: (B, T_text) — frames per token
    """
    B, T_spec = alignment.shape
    T_text = token_mask.shape[1] if token_mask is not None else int(alignment.max().item()) + 1

    device = alignment.device
    durations = torch.zeros(B, T_text, device=device)
    ones = torch.ones_like(alignment, dtype=durations.dtype, device=device)
    durations.scatter_add_(1, alignment, ones)

    if token_mask is not None:
        durations = durations.masked_fill((durations < min_duration) & ~token_mask, min_duration)

    return durations.long()
