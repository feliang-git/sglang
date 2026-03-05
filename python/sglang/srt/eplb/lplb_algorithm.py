import logging

import numpy as np
import torch
from scipy.optimize import linprog

logger = logging.getLogger(__name__)


def lplb_algorithm(
    phy2log: torch.Tensor,
    logcnt: torch.Tensor,
    log2phy: torch.Tensor,
    g: int,
    logical_count: torch.Tensor,
    device: torch.device,
):
    """Solve LP to compute load-balanced dispatch probabilities for redundant experts.

    Formulates a min-max LP per layer: minimize the maximum GPU load by
    optimally splitting each replicated expert's token share across its
    physical replicas. Layers with identical logcnt share one LP structure.

    Returns:
        log2phy_prob: (layers, num_logical_experts, X) probability tensor
    """
    num_layers = phy2log.shape[0]
    log2phy_prob = torch.zeros(log2phy.shape, dtype=torch.float32, device=device)

    num_phy: int = phy2log.shape[1]
    num_phy_gpu: int = num_phy // g
    gpu_ids = torch.arange(num_phy, device=device) // num_phy_gpu

    lc_all = logical_count.to(device=device, dtype=torch.float32)
    lc_sums = lc_all.sum(dim=1, keepdim=True)
    zero_layers = lc_sums.squeeze(1) == 0
    lc_norm = lc_all / lc_sums.clamp(min=1e-30)

    # Group layers by logcnt signature -- same logcnt means identical
    # A matrix structure (same NC, NV, same expert classification).
    logcnt_cpu = logcnt.cpu()
    groups: dict[bytes, list[int]] = {}
    for layer_id in range(num_layers):
        key = logcnt_cpu[layer_id].numpy().tobytes()
        groups.setdefault(key, []).append(layer_id)

    for _, layer_ids in groups.items():
        _solve_group(
            layer_ids, phy2log, logcnt, log2phy, g,
            lc_norm, zero_layers, gpu_ids, num_phy,
            log2phy_prob, device,
        )

    return log2phy_prob


def _solve_group(
    layer_ids: list[int],
    phy2log: torch.Tensor,
    logcnt: torch.Tensor,
    log2phy: torch.Tensor,
    g: int,
    lc_norm: torch.Tensor,
    zero_layers: torch.Tensor,
    gpu_ids: torch.Tensor,
    num_phy: int,
    log2phy_prob: torch.Tensor,
    device: torch.device,
):
    """Solve LP for a group of layers sharing the same expert placement structure."""
    ref = layer_ids[0]
    layer_phy2log = phy2log[ref].to(device)
    layer_logcnt = logcnt[ref].to(device)
    layer_log2phy = log2phy[ref].to(device)

    # Classify experts: single-copy (deterministic) vs replicated (LP decides split)
    log_single = torch.nonzero(layer_logcnt == 1).flatten()
    phy_single = layer_log2phy[log_single, 0]
    log_rep = torch.nonzero(layer_logcnt > 1).flatten()
    phy_rep = torch.nonzero(layer_logcnt[layer_phy2log] > 1).flatten()

    n_single = len(log_single)
    n_log_rep = len(log_rep)
    n_phy_rep = len(phy_rep)

    if n_log_rep == 0:
        for lid in layer_ids:
            if zero_layers[lid]:
                log2phy_prob[lid] = (log2phy[lid].to(device) >= 0).to(torch.float32)
            else:
                _postprocess_single_only(
                    lid, log_single, phy_single, log2phy[lid].to(device),
                    lc_norm, log2phy_prob, device,
                )
        return

    # GPU membership for single/replicated physical experts
    B1_gpu = gpu_ids[phy_single]
    B2_gpu = gpu_ids[phy_rep]

    # Map replicated physical experts to their logical expert row index
    phy2log_rep = layer_phy2log[phy_rep]
    log_rep_index = torch.empty(int(layer_logcnt.shape[0]), dtype=torch.long, device=device)
    log_rep_index[log_rep] = torch.arange(n_log_rep, device=device)
    c_row_indices = log_rep_index[phy2log_rep]
    c_col_indices = torch.arange(n_phy_rep, device=device)

    NC = n_log_rep + g
    NV = n_phy_rep + g + 1

    # A = [[C, 0, 0], [B2, I, -1]] where:
    #   C: replica-to-logical mapping (prob sums = token share)
    #   B2: GPU membership (load balance), I: slack, -1: max-load variable
    A_template = torch.zeros((NC, NV), dtype=torch.float32, device=device)
    A_template[c_row_indices, c_col_indices] = 1.0
    A_template[n_log_rep + B2_gpu, c_col_indices] = 1.0
    bottom_rows = torch.arange(g, device=device)
    A_template[n_log_rep + bottom_rows, n_phy_rep + bottom_rows] = 1.0
    A_template[n_log_rep:, n_phy_rep + g] = -1.0

    A_base_rowsum = A_template.sum(dim=1)

    # Objective: minimize max-load (c[-2]=1) with big-M penalty (c[-1]=1000)
    c_vec = np.zeros(NV + 1, dtype=np.float64)
    c_vec[-2] = 1.0
    c_vec[-1] = 1000.0
    A_base_np = A_template.cpu().numpy().astype(np.float64)

    layer_log2phy_safe = layer_log2phy.clone()
    layer_log2phy_safe[layer_log2phy_safe < 0] = 0
    invalid_mask = layer_log2phy < 0

    n_group = len(layer_ids)
    lid_tensor = torch.tensor(layer_ids, dtype=torch.long)

    # b = [b1, b2]: b1 = token share per replicated expert,
    # b2 = -(single-expert load per GPU)
    t1_all = lc_norm[lid_tensor][:, log_single]
    b1_all = lc_norm[lid_tensor][:, log_rep]

    b2_all = torch.zeros((n_group, g), dtype=torch.float32, device=device)
    for i in range(n_group):
        b2_all[i].scatter_add_(0, B1_gpu, t1_all[i])
    b2_all = -b2_all

    b_all = torch.cat([b1_all, b2_all], dim=1)
    # Big-M artificial variable column for LP feasibility
    bigM_all = b_all - A_base_rowsum.unsqueeze(0)

    b_all_np = b_all.cpu().numpy().astype(np.float64)
    bigM_all_np = bigM_all.cpu().numpy().astype(np.float64)

    results_np = np.zeros((n_group, NV + 1), dtype=np.float64)
    solve_success = np.ones(n_group, dtype=bool)

    A_np = np.empty((NC, NV + 1), dtype=np.float64)
    A_np[:, :NV] = A_base_np

    for i, lid in enumerate(layer_ids):
        if zero_layers[lid]:
            solve_success[i] = False
            continue

        A_np[:, NV] = bigM_all_np[i]
        res = linprog(c_vec, A_eq=A_np, b_eq=b_all_np[i],
                      bounds=(0, None), method="highs")

        if res.success:
            results_np[i] = res.x
        else:
            solve_success[i] = False

    n_fallback = int((~solve_success).sum())
    if n_fallback > 0:
        logger.warning(
            f"[lplb] {n_fallback}/{len(layer_ids)} layers in group "
            f"(NC={NC},NV={NV}) fell back to uniform"
        )

    results_gpu = torch.from_numpy(results_np[:, :n_phy_rep]).to(
        device=device, dtype=torch.float32
    ).clamp(min=0.0)

    fallback = (layer_log2phy >= 0).to(torch.float32)

    # Convert LP solution vector to per-expert probability matrix
    for i, lid in enumerate(layer_ids):
        if not solve_success[i]:
            log2phy_prob[lid] = fallback
            continue

        phy_prob = torch.zeros(
            n_single + n_phy_rep + 1, dtype=torch.float32, device=device,
        )
        phy_prob[phy_rep] = results_gpu[i]
        phy_prob[phy_single] = t1_all[i]
        prob = torch.take(phy_prob, layer_log2phy_safe)
        log2phy_prob[lid] = prob.masked_fill(invalid_mask, 0)


def _postprocess_single_only(
    layer_id: int,
    log_single: torch.Tensor,
    phy_single: torch.Tensor,
    layer_log2phy: torch.Tensor,
    lc_norm: torch.Tensor,
    log2phy_prob: torch.Tensor,
    device: torch.device,
):
    """Handle layers where all experts are single-copy (no LP needed)."""
    t1 = lc_norm[layer_id][log_single]
    n_single = len(log_single)
    phy_prob = torch.zeros(n_single + 1, dtype=torch.float32, device=device)
    phy_prob[phy_single] = t1
    safe = layer_log2phy.clone()
    safe[safe < 0] = 0
    prob = torch.take(phy_prob, safe)
    log2phy_prob[layer_id] = prob.masked_fill(layer_log2phy < 0, 0)


def single_layer_lplb_algorithm(
    layer_phy2log: torch.Tensor,
    layer_logcnt: torch.Tensor,
    layer_log2phy: torch.Tensor,
    g: int,
    logical_count: torch.Tensor,
    device: torch.device,
):
    return lplb_algorithm(
        layer_phy2log.unsqueeze(0),
        layer_logcnt.unsqueeze(0),
        layer_log2phy.unsqueeze(0),
        g,
        logical_count.unsqueeze(0),
        device,
    )[0]
