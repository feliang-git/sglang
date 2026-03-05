"""
Analyze LPLB diagnostic data saved during server run.

Usage:
    python tests/analyze_lplb_diagnostics.py /workspace/lplb_diagnostics.pt
"""

import sys

import numpy as np
import torch


def analyze(path: str):
    data = torch.load(path, weights_only=False, map_location="cpu")

    g = data["g"]
    num_groups = data["num_groups"]
    total_ms = data["total_ms"]
    log2phy_prob = data["log2phy_prob"]
    logcnt = data["logcnt"]
    log2phy = data["log2phy"]
    logical_count = data["logical_count"]

    num_layers = log2phy_prob.shape[0]
    num_log = log2phy_prob.shape[1]
    num_slots = log2phy_prob.shape[2]

    print(f"{'='*72}")
    print(f"  LPLB Diagnostics Report")
    print(f"{'='*72}")
    print(f"  Layers:       {num_layers}")
    print(f"  Logical exp:  {num_log}")
    print(f"  Slots:        {num_slots}")
    print(f"  GPUs:         {g}")
    print(f"  Groups:       {num_groups}")
    print(f"  Total time:   {total_ms:.1f} ms")
    print()

    # ---- Per-layer LP analysis ----
    print(f"{'='*72}")
    print(f"  Per-Layer LP Analysis")
    print(f"{'='*72}")

    fallback_layers = []
    bad_residual_layers = []
    bad_bigm_layers = []

    for lid in range(num_layers):
        key = f"layer_{lid}"
        if key not in data:
            continue
        ld = data[key]

        if not ld["solve_success"]:
            fallback_layers.append(lid)
            continue

        res = ld.get("max_constraint_residual", 0)
        bigm = ld.get("big_m_var", 0)
        obj = ld.get("objective_value", 0)
        xmin = ld.get("x_min", 0)

        if res > 1e-6:
            bad_residual_layers.append((lid, res))
        if abs(bigm) > 1e-6:
            bad_bigm_layers.append((lid, bigm))

    print(f"\n  Fallback (uniform) layers: {len(fallback_layers)}/{num_layers}")
    if fallback_layers:
        print(f"    Layer IDs: {fallback_layers[:20]}{'...' if len(fallback_layers) > 20 else ''}")

    print(f"  Layers with constraint residual > 1e-6: {len(bad_residual_layers)}")
    for lid, res in bad_residual_layers[:5]:
        print(f"    Layer {lid}: residual = {res:.6e}")

    print(f"  Layers with big-M > 1e-6: {len(bad_bigm_layers)}")
    for lid, bigm in bad_bigm_layers[:5]:
        print(f"    Layer {lid}: big-M = {bigm:.6e}")

    # Collect objective values and residuals for successful layers
    objectives = []
    residuals = []
    for lid in range(num_layers):
        key = f"layer_{lid}"
        if key in data and data[key]["solve_success"]:
            objectives.append(data[key]["objective_value"])
            residuals.append(data[key]["max_constraint_residual"])

    if objectives:
        print(f"\n  Objective values (successful layers):")
        print(f"    min:  {min(objectives):.8f}")
        print(f"    max:  {max(objectives):.8f}")
        print(f"    mean: {np.mean(objectives):.8f}")

        print(f"\n  Constraint residuals:")
        print(f"    max:  {max(residuals):.6e}")
        print(f"    mean: {np.mean(residuals):.6e}")

    # ---- Final probability analysis ----
    print(f"\n{'='*72}")
    print(f"  Final Probability Analysis (log2phy_prob)")
    print(f"{'='*72}")

    has_neg = (log2phy_prob < -1e-8).any().item()
    print(f"\n  Non-negativity:    {'PASS' if not has_neg else 'FAIL'} (min={log2phy_prob.min().item():.6e})")

    valid_mask = log2phy >= 0
    invalid_nonzero = ((~valid_mask) & (log2phy_prob.abs() > 1e-10)).sum().item()
    print(f"  Invalid pos zero:  {'PASS' if invalid_nonzero == 0 else 'FAIL'} ({int(invalid_nonzero)} violations)")

    # Row-sum check (exclude fallback layers since they return all-ones
    # which get normalized by the outer wrapper)
    lc_norm = logical_count.float()
    lc_sums = lc_norm.sum(dim=1, keepdim=True)
    lc_norm = lc_norm / lc_sums.clamp(min=1e-30)

    row_sums = log2phy_prob.sum(dim=-1)

    solved_mask = torch.ones(num_layers, dtype=torch.bool)
    for lid in fallback_layers:
        solved_mask[lid] = False

    if solved_mask.any():
        solved_dev = (row_sums[solved_mask] - lc_norm[solved_mask]).abs()
        max_dev = solved_dev.max().item()
        mean_dev = solved_dev.mean().item()
    else:
        max_dev = 0.0
        mean_dev = 0.0
    print(f"  Row-sum conservation (solved layers only):")
    print(f"    max_dev={max_dev:.6e}, mean_dev={mean_dev:.6e}")
    print(f"    {'PASS' if max_dev < 1e-4 else 'FAIL'}")

    if fallback_layers:
        fb_row_sums = row_sums[torch.tensor(fallback_layers)]
        print(f"  Fallback layers row-sum (pre-normalization):")
        print(f"    range: [{fb_row_sums.min().item():.4f}, {fb_row_sums.max().item():.4f}]")
        print(f"    (these get normalized by outer wrapper -- expected to be > 1.0)")

    # Zero rows (experts with tokens)
    has_tokens = lc_norm > 1e-12
    zero_with_tokens = ((row_sums < 1e-10) & has_tokens).sum().item()
    total_with_tokens = has_tokens.sum().item()
    print(f"  Zero rows (with tokens): {int(zero_with_tokens)}/{int(total_with_tokens)}")
    print(f"    {'PASS' if zero_with_tokens == 0 else 'FAIL'}")

    # Per-layer NC/NV distribution
    print(f"\n{'='*72}")
    print(f"  LP Dimension Distribution")
    print(f"{'='*72}")

    dims = {}
    for lid in range(num_layers):
        key = f"layer_{lid}"
        if key in data:
            nc = data[key]["NC"]
            nv = data[key]["NV"]
            dims.setdefault((nc, nv), []).append(lid)

    print(f"\n  {'(NC, NV)':<15} {'Count':<8} {'Layers (sample)'}")
    for (nc, nv), lids in sorted(dims.items()):
        sample = lids[:5]
        print(f"  ({nc:3d}, {nv:3d})     {len(lids):<8} {sample}")

    # Overall verdict
    print(f"\n{'='*72}")
    all_ok = (
        not has_neg
        and invalid_nonzero == 0
        and max_dev < 1e-4
        and zero_with_tokens == 0
        and len(bad_residual_layers) == 0
        and len(bad_bigm_layers) == 0
    )
    print(f"  OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    print(f"{'='*72}")

    return all_ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <diagnostics.pt>")
        sys.exit(1)
    ok = analyze(sys.argv[1])
    sys.exit(0 if ok else 1)
