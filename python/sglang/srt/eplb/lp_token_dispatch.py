import numpy as np
import torch
import torch.distributed as dist
import time

from sglang.srt.distributed import get_world_group, get_moe_ep_group
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from lplb.minilp import solve_ipm


def run_lp_solver(
    global_counts: torch.Tensor,
    A: torch.Tensor,
    B1: torch.Tensor,
    B2: torch.Tensor,
    c: torch.Tensor,
    log_single_expert_array: torch.Tensor,
    phy_single_expert_array: torch.Tensor,
    log_replicated_expert_array: torch.Tensor,
    phy_replicated_expert_array: torch.Tensor,
    log2phy: torch.Tensor,
):
    global_counts = global_counts.to(torch.float32)
    global_counts = global_counts / global_counts.sum()
    device = global_counts.device
    t1: torch.Tensor = global_counts[log_single_expert_array]
    left = B1 @ t1
    b2 = -left.flatten()
    b1 = global_counts[log_replicated_expert_array].to(torch.float32)
    b = torch.cat([b1, b2])

    big_M_col = b - torch.sum(A, dim=1)
    A = torch.hstack([A, big_M_col.reshape(-1, 1)])

    avail_counter = torch.zeros((), dtype=torch.int, device=device)

    result = solve_ipm(
        A,
        b,
        c,
        avail_counter,
        None,
    )

    x = result[:phy_replicated_expert_array.shape[0]]
    x[x < 0.0] = 0.0
    phy_prob = torch.zeros(
        log_single_expert_array.shape[0]
        + phy_replicated_expert_array.shape[0]
        + 1,
        dtype=torch.float32,
        device=device,
    )
    phy_prob[phy_replicated_expert_array] = x
    phy_prob[phy_single_expert_array] = t1
    log2phy_prob = torch.zeros(log2phy.shape, dtype=torch.float32, device=device)
    log2phy_prob = torch.take(phy_prob, log2phy)

    return log2phy_prob


def count_logical_expert_tokens(
    logical_expert_ids: torch.Tensor, num_logical_experts: int
) -> torch.Tensor:
    """Count logical expert token occurrences from topk selection

    Args:
        logical_expert_ids: Tensor of shape (num_tokens, topk) containing logical expert IDs
        num_logical_experts: Number of logical experts

    Returns:
        Tensor of shape (num_logical_experts,) containing token counts for each expert
    """
    device = logical_expert_ids.device
    logical_counts = torch.zeros(num_logical_experts, dtype=torch.int32, device=device)

    flat_ids = logical_expert_ids.flatten()
    logical_counts.scatter_add_(
        dim=0,
        index=flat_ids.long(),
        src=torch.ones_like(flat_ids, dtype=torch.int32),
    )

    return logical_counts


def get_global_logical_counts_on_rank0(local_counts: torch.Tensor) -> torch.Tensor:
    """Get global logical counts using SGLang's parallel state system.

    All ranks move local_counts to CPU, then use the CPU communication group for reduce.
    The result is only correct on rank 0.

    Args:
        local_counts: Local logical counts tensor (on GPU)

    Returns:
        Global logical counts tensor on GPU
    """
    group = get_moe_ep_group()

    if group.world_size == 1:
        # Single rank case, just return local counts
        return local_counts

    group.all_reduce(local_counts)
    return local_counts


def send_log2phy_prob_broadcast(log2phy_prob: torch.Tensor):
    """Send log2phy_prob to all ranks"""
    group = get_moe_ep_group()
    group.broadcast(log2phy_prob, src=0)
    return log2phy_prob


def get_log2phy_prob(
    topk_ids: torch.Tensor,
    expert_location_dispatch_info: ExpertLocationDispatchInfo,
):
    """Using Linear Programming to get the redundant token distribution probability

    Args:
        topk_ids: Tensor of shape (num_tokens, topk) containing logical expert IDs
        num_logical_experts: Number of logical experts

    Returns:
        Tensor of shape (num_logical_experts,) containing global token counts for each expert
    """
    device = topk_ids.device
    num_logical_experts = (
        expert_location_dispatch_info.partial_logical_to_all_physical_map_num_valid.shape[
            0
        ]
    )
    # Step 1: Count local logical expert tokens
    local_counts = count_logical_expert_tokens(topk_ids, num_logical_experts)

    # Step 2: All-reduce to get global counts
    global_counts = get_global_logical_counts_on_rank0(local_counts)

    # Step 3: Use LP to get the redundant token distribution probability
    if dist.get_rank() == 0:
        log2phy_prob = run_lp_solver(
            global_counts,
            expert_location_dispatch_info.A,
            expert_location_dispatch_info.B1,
            expert_location_dispatch_info.B2,
            expert_location_dispatch_info.c,
            expert_location_dispatch_info.log_single_expert_array,
            expert_location_dispatch_info.phy_single_expert_array,
            expert_location_dispatch_info.log_replicated_expert_array,
            expert_location_dispatch_info.phy_replicated_expert_array,
            expert_location_dispatch_info.partial_logical_to_all_physical_map,
        )
    else:
        log2phy_prob = torch.zeros_like(
            expert_location_dispatch_info.partial_logical_to_all_physical_map,
            device=device,
        ).to(torch.float32)

    # Step 4: Broadcast to all ranks
    log2phy_prob = send_log2phy_prob_broadcast(log2phy_prob)

    return log2phy_prob
