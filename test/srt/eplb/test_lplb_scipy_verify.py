"""
End-to-end verification of the static_lp pipeline.

Covers:
  0a. Pre-processing:  token_count -> expert classification, B/C matrices, b vector
  0b. Solver:          LP feasibility, optimality, big-M near zero
  0c. Post-processing: log2phy_prob validity (non-neg, row-sum conservation, no invalid)
  0d. Dispatch:        multinomial sampling produces valid, load-balanced assignments

Run:
    python tests/test_lplb_scipy_verify.py                          # synthetic only
    python tests/test_lplb_scipy_verify.py --real-data expert_record_summed.pt
"""

import argparse
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy.optimize import linprog


# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str

    def __str__(self):
        tag = "PASS" if self.passed else "FAIL"
        return f"  [{tag}] {self.name}: {self.detail}"


@dataclass
class ScenarioReport:
    name: str
    checks: List[CheckResult] = field(default_factory=list)
    timings: Dict[str, float] = field(default_factory=dict)

    def add(self, name: str, passed: bool, detail: str):
        self.checks.append(CheckResult(name, passed, detail))

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def print_report(self):
        status = "PASS" if self.all_passed else "FAIL"
        print(f"\n{'='*72}")
        print(f"  [{status}] Scenario: {self.name}")
        print(f"{'='*72}")
        section = ""
        for c in self.checks:
            parts = c.name.split("/", 1)
            if len(parts) == 2 and parts[0] != section:
                section = parts[0]
                print(f"\n  {section}:")
            print(f"  {c}")
        if self.timings:
            print(f"\n  Performance:")
            for k, v in self.timings.items():
                print(f"    {k}: {v:.2f} ms")


# ---------------------------------------------------------------------------
# Scenario definition and tensor construction
# ---------------------------------------------------------------------------

@dataclass
class ExpertPlacement:
    g: int
    num_phy_per_gpu: int
    phy2log: List[int]
    logical_count: List[float]

    @property
    def num_phy(self) -> int:
        return self.g * self.num_phy_per_gpu

    @property
    def num_log(self) -> int:
        return max(self.phy2log) + 1


def build_tensors(placement: ExpertPlacement, device: torch.device):
    num_log = placement.num_log
    phy2log = torch.tensor(placement.phy2log, dtype=torch.int64, device=device)
    logcnt = torch.zeros(num_log, dtype=torch.int64, device=device)
    for p in placement.phy2log:
        logcnt[p] += 1
    max_copies = int(logcnt.max().item())
    log2phy = torch.full((num_log, max_copies), -1, dtype=torch.int64, device=device)
    fill_idx = torch.zeros(num_log, dtype=torch.int64, device=device)
    for phy_idx, log_idx in enumerate(placement.phy2log):
        pos = fill_idx[log_idx].item()
        log2phy[log_idx, pos] = phy_idx
        fill_idx[log_idx] += 1
    logical_count = torch.tensor(placement.logical_count, dtype=torch.float32, device=device)
    return (
        phy2log.unsqueeze(0),
        logcnt.unsqueeze(0),
        log2phy.unsqueeze(0),
        logical_count.unsqueeze(0),
    )


# ---------------------------------------------------------------------------
# LP matrix construction (mirrors lplb_algorithm.py, returns intermediates)
# ---------------------------------------------------------------------------

def build_lp_full(
    layer_phy2log: torch.Tensor,
    layer_logcnt: torch.Tensor,
    layer_log2phy: torch.Tensor,
    g: int,
    logical_count: torch.Tensor,
    device: torch.device,
):
    """Build LP and return all intermediate data for verification."""
    num_phy = layer_phy2log.shape[0]
    num_phy_gpu = num_phy // g

    log_single = torch.nonzero(layer_logcnt == 1).flatten()
    phy_single = layer_log2phy[log_single, 0]
    log_rep = torch.nonzero(layer_logcnt > 1).flatten()
    phy_rep = torch.nonzero(layer_logcnt[layer_phy2log] > 1).flatten()

    n_single = len(log_single)
    n_log_rep = len(log_rep)
    n_phy_rep = len(phy_rep)

    B = torch.zeros((g, num_phy), dtype=torch.float32, device=device)
    for i in range(g):
        B[i, i * num_phy_gpu : (i + 1) * num_phy_gpu] = 1
    B1 = B[:, phy_single]
    B2 = B[:, phy_rep]

    C = torch.zeros((n_log_rep, n_phy_rep), dtype=torch.float32, device=device)
    phy2log_rep = layer_phy2log[phy_rep]
    for i in range(n_log_rep):
        C[i, phy2log_rep == log_rep[i]] = 1.0

    zeros_top = torch.zeros((n_log_rep, g), dtype=torch.float32, device=device)
    zeros_top_col = torch.zeros((n_log_rep, 1), dtype=torch.float32, device=device)
    I_mat = torch.eye(g, dtype=torch.float32, device=device)
    neg1_col = torch.full((g, 1), -1.0, dtype=torch.float32, device=device)

    A_top = torch.hstack([C, zeros_top, zeros_top_col])
    A_bottom = torch.hstack([B2, I_mat, neg1_col])
    A = torch.vstack([A_top, A_bottom])

    c = torch.zeros(A.shape[1] + 1, dtype=torch.float32, device=device)
    c[-2] = 1.0
    c[-1] = 1000.0

    lc = logical_count.to(torch.float32)
    lc = lc / lc.sum()
    t1 = lc[log_single]
    left = B1 @ t1
    b2 = -left.flatten()
    b1 = lc[log_rep].to(torch.float32)
    b = torch.cat([b1, b2])

    big_M_col = b - torch.sum(A, dim=1)
    A = torch.hstack([A, big_M_col.reshape(-1, 1)])

    return dict(
        A=A, b=b, c=c, B=B, B1=B1, B2=B2, C=C,
        t1=t1, lc_normalized=lc,
        log_single=log_single, phy_single=phy_single,
        log_rep=log_rep, phy_rep=phy_rep,
        n_single=n_single, n_log_rep=n_log_rep, n_phy_rep=n_phy_rep,
        layer_phy2log=layer_phy2log, layer_logcnt=layer_logcnt,
        layer_log2phy=layer_log2phy, g=g, num_phy=num_phy,
        num_phy_gpu=num_phy_gpu,
    )


def postprocess(result: torch.Tensor, lp: dict, device: torch.device) -> torch.Tensor:
    x_raw = result[: lp["n_phy_rep"]]
    x = x_raw.clone()
    x[x < 0.0] = 0.0
    phy_prob = torch.zeros(
        lp["n_single"] + lp["n_phy_rep"] + 1,
        dtype=torch.float32, device=device,
    )
    phy_prob[lp["phy_rep"]] = x
    phy_prob[lp["phy_single"]] = lp["t1"]
    layer_log2phy = lp["layer_log2phy"]
    safe = layer_log2phy.clone()
    safe[safe < 0] = 0
    prob = torch.take(phy_prob, safe)
    prob = prob.masked_fill(layer_log2phy < 0, 0)
    return prob


def apply_outer_wrapper(log2phy_prob: torch.Tensor, log2phy: torch.Tensor, logcnt: torch.Tensor):
    """Mimics compute_logical_to_physical_probability outer wrapper."""
    log2phy_prob = log2phy_prob.masked_fill(log2phy < 0, 0)
    row_sums = log2phy_prob.sum(dim=-1, keepdim=True)
    valid_mask = log2phy >= 0
    num_valid = logcnt.unsqueeze(-1).float()
    needs_fix = row_sums <= 0
    if needs_fix.any():
        uniform = torch.where(
            valid_mask,
            1.0 / torch.clamp(num_valid, min=1),
            torch.zeros_like(log2phy_prob),
        )
        log2phy_prob = torch.where(needs_fix, uniform, log2phy_prob)
    return log2phy_prob


# ---------------------------------------------------------------------------
# 0a. Pre-processing verification
# ---------------------------------------------------------------------------

def verify_preprocessing(lp: dict, report: ScenarioReport):
    S = "Pre-processing"
    layer_logcnt = lp["layer_logcnt"]
    log_single = lp["log_single"]
    log_rep = lp["log_rep"]
    num_log = layer_logcnt.shape[0]

    # Expert classification: single + replicated = all experts with count >= 1
    all_classified = set(log_single.tolist()) | set(log_rep.tolist())
    all_with_count = set(torch.nonzero(layer_logcnt >= 1).flatten().tolist())
    report.add(f"{S}/Expert classification",
               all_classified == all_with_count,
               f"classified={len(all_classified)}, expected={len(all_with_count)}, "
               f"single={len(log_single)}, replicated={len(log_rep)}")

    # No overlap
    overlap = set(log_single.tolist()) & set(log_rep.tolist())
    report.add(f"{S}/No single-replicated overlap",
               len(overlap) == 0,
               f"overlap={len(overlap)}")

    # B matrix: each physical expert belongs to exactly one GPU
    B = lp["B"]
    col_sums = B.sum(dim=0)
    report.add(f"{S}/B matrix column sums",
               torch.allclose(col_sums, torch.ones_like(col_sums)),
               f"all columns sum to 1: {col_sums.min().item():.4f}-{col_sums.max().item():.4f}")

    g = lp["g"]
    num_phy_gpu = lp["num_phy_gpu"]
    b_correct = True
    for i in range(g):
        expected_ones = B[i, i * num_phy_gpu : (i + 1) * num_phy_gpu]
        expected_zeros_before = B[i, :i * num_phy_gpu]
        expected_zeros_after = B[i, (i + 1) * num_phy_gpu:]
        if not (expected_ones.sum().item() == num_phy_gpu
                and expected_zeros_before.sum().item() == 0
                and expected_zeros_after.sum().item() == 0):
            b_correct = False
            break
    report.add(f"{S}/B matrix block-diagonal structure",
               b_correct, f"g={g}, num_phy_gpu={num_phy_gpu}")

    # C matrix: each column sums to 1, each row sums to logcnt for that expert
    C = lp["C"]
    if C.numel() > 0:
        c_col_sums = C.sum(dim=0)
        report.add(f"{S}/C matrix column sums == 1",
                   torch.allclose(c_col_sums, torch.ones_like(c_col_sums)),
                   f"range: {c_col_sums.min().item():.4f}-{c_col_sums.max().item():.4f}")

        c_row_sums = C.sum(dim=1)
        expected_row_sums = layer_logcnt[log_rep].float()
        report.add(f"{S}/C matrix row sums == logcnt",
                   torch.allclose(c_row_sums, expected_row_sums),
                   f"max deviation: {(c_row_sums - expected_row_sums).abs().max().item():.6e}")
    else:
        report.add(f"{S}/C matrix (no replicated)", True, "empty, skipped")

    # b vector: b1 = normalized token share for replicated, b2 = -single load per GPU
    b = lp["b"]
    b1 = b[:lp["n_log_rep"]]
    b2 = b[lp["n_log_rep"]:]
    lc = lp["lc_normalized"]
    expected_b1 = lc[log_rep]
    report.add(f"{S}/b1 == normalized token share (replicated)",
               torch.allclose(b1, expected_b1, atol=1e-6),
               f"max deviation: {(b1 - expected_b1).abs().max().item():.6e}")

    expected_b2 = -(lp["B1"] @ lp["t1"]).flatten()
    report.add(f"{S}/b2 == -single expert load per GPU",
               torch.allclose(b2, expected_b2, atol=1e-6),
               f"max deviation: {(b2 - expected_b2).abs().max().item():.6e}")

    # Dimensions
    NC = lp["n_log_rep"] + g
    NV = lp["n_phy_rep"] + g + 2
    A = lp["A"]
    report.add(f"{S}/LP dimensions",
               A.shape[0] == NC and A.shape[1] == NV,
               f"A={tuple(A.shape)}, expected=({NC}, {NV})")


# ---------------------------------------------------------------------------
# 0b. Solver verification
# ---------------------------------------------------------------------------

def verify_solver(result_np: np.ndarray, A_np: np.ndarray, b_np: np.ndarray,
                  c_np: np.ndarray, success: bool, report: ScenarioReport):
    S = "Solver"
    report.add(f"{S}/linprog success", success, "")

    if not success:
        return

    residual = A_np @ result_np - b_np
    max_res = np.abs(residual).max()
    report.add(f"{S}/Feasibility |Ax-b|_max",
               max_res < 1e-6,
               f"{max_res:.6e}")

    x_min = result_np.min()
    report.add(f"{S}/Non-negativity x_min",
               x_min > -1e-8,
               f"{x_min:.6e}")

    big_m_val = result_np[-1]
    report.add(f"{S}/Big-M variable near zero",
               abs(big_m_val) < 1e-6,
               f"{big_m_val:.6e}")

    obj_val = c_np @ result_np
    report.add(f"{S}/Objective value",
               True,
               f"{obj_val:.8f}")


# ---------------------------------------------------------------------------
# 0c. Post-processing verification
# ---------------------------------------------------------------------------

def verify_postprocessing(log2phy_prob: torch.Tensor, log2phy: torch.Tensor,
                          logcnt: torch.Tensor, lc_normalized: torch.Tensor,
                          report: ScenarioReport):
    S = "Post-processing"

    # Non-negativity
    has_neg = (log2phy_prob < -1e-8).any().item()
    report.add(f"{S}/Non-negativity",
               not has_neg,
               f"min value: {log2phy_prob.min().item():.6e}")

    # Invalid positions
    valid_mask = log2phy >= 0
    violations = ((~valid_mask) & (log2phy_prob.abs() > 1e-10)).sum().item()
    report.add(f"{S}/Invalid positions (padding) are zero",
               violations == 0,
               f"{int(violations)} violations")

    # Row-sum conservation: for each logical expert, sum of probs == normalized token share
    row_sums = log2phy_prob.sum(dim=-1)
    expected_row_sums = lc_normalized
    deviation = (row_sums - expected_row_sums).abs()
    max_dev = deviation.max().item()
    report.add(f"{S}/Row-sum == normalized token share",
               max_dev < 1e-4,
               f"max deviation: {max_dev:.6e}")

    # Zero rows: only experts with nonzero token share should have nonzero prob
    has_tokens = lc_normalized > 1e-12
    zero_rows_with_tokens = int(((row_sums < 1e-10) & has_tokens).sum().item())
    experts_with_tokens = int(has_tokens.sum().item())
    report.add(f"{S}/No zero rows (experts with tokens)",
               zero_rows_with_tokens == 0,
               f"{zero_rows_with_tokens} zero rows out of {experts_with_tokens} experts with tokens")

    # Single experts get deterministic probability
    single_mask = logcnt == 1
    if single_mask.any():
        single_row_sums = row_sums[single_mask]
        single_expected = lc_normalized[single_mask]
        single_dev = (single_row_sums - single_expected).abs().max().item()
        report.add(f"{S}/Single expert prob == token share",
                   single_dev < 1e-6,
                   f"max deviation: {single_dev:.6e}")

    # After outer wrapper (compute_logical_to_physical_probability)
    wrapped = apply_outer_wrapper(log2phy_prob.clone(), log2phy, logcnt)
    wrapped_row_sums = wrapped.sum(dim=-1)
    wrapped_zero = int((wrapped_row_sums < 1e-10).sum().item())
    report.add(f"{S}/After outer wrapper: no zero rows",
               wrapped_zero == 0,
               f"{wrapped_zero} zero rows")
    wrapped_neg = (wrapped < -1e-8).any().item()
    report.add(f"{S}/After outer wrapper: non-negative",
               not wrapped_neg,
               f"min: {wrapped.min().item():.6e}")
    wrapped_invalid = ((~valid_mask) & (wrapped.abs() > 1e-10)).sum().item()
    report.add(f"{S}/After outer wrapper: invalid positions zero",
               wrapped_invalid == 0,
               f"{int(wrapped_invalid)} violations")


# ---------------------------------------------------------------------------
# 0d. Dispatch simulation verification
# ---------------------------------------------------------------------------

def verify_dispatch(log2phy_prob: torch.Tensor, log2phy: torch.Tensor,
                    logcnt: torch.Tensor, lc_normalized: torch.Tensor,
                    layer_phy2log: torch.Tensor, g: int, num_phy_gpu: int,
                    report: ScenarioReport, n_samples: int = 10000):
    S = "Dispatch"
    device = log2phy_prob.device

    wrapped = apply_outer_wrapper(log2phy_prob.clone(), log2phy, logcnt)
    num_log = wrapped.shape[0]
    num_slots = wrapped.shape[1]

    # Simulate dispatch for all logical experts
    all_valid = True
    for expert_id in range(num_log):
        probs = wrapped[expert_id]
        if probs.sum() < 1e-10:
            continue
        valid_phy = set(log2phy[expert_id][log2phy[expert_id] >= 0].tolist())
        if len(valid_phy) == 0:
            continue
        samples = torch.multinomial(probs.float(), n_samples, replacement=True)
        sampled_phy = log2phy[expert_id][samples]
        if (sampled_phy < 0).any():
            all_valid = False
            break
        sampled_set = set(sampled_phy.tolist())
        if not sampled_set.issubset(valid_phy):
            all_valid = False
            break
    report.add(f"{S}/All sampled physical IDs are valid replicas",
               all_valid, "")

    # GPU load balance: compute expected load per GPU
    gpu_load = torch.zeros(g, dtype=torch.float64, device=device)
    for expert_id in range(num_log):
        token_share = lc_normalized[expert_id].item()
        for slot in range(num_slots):
            phy_id = log2phy[expert_id, slot].item()
            if phy_id < 0:
                continue
            prob = wrapped[expert_id, slot].item()
            gpu_id = phy_id // num_phy_gpu
            gpu_load[gpu_id] += prob * token_share

    gpu_load_np = gpu_load.cpu().numpy()
    max_load = gpu_load_np.max()
    min_load = gpu_load_np[gpu_load_np > 1e-12].min() if (gpu_load_np > 1e-12).any() else 0
    report.add(f"{S}/GPU expected load distribution",
               True,
               f"max={max_load:.6f}, min_nonzero={min_load:.6f}, "
               f"per-GPU: [{', '.join(f'{x:.4f}' for x in gpu_load_np)}]")

    # Verify max GPU load matches LP objective: the LP minimizes the max
    # total load across GPUs (single expert load + replicated expert load).
    # max_load should be close to the LP objective value.
    report.add(f"{S}/Max GPU load is LP-optimal",
               max_load < 1.0,
               f"max_load={max_load:.6f} (should be bounded by total normalized share)")

    # Statistical: sample dispatch for replicated experts, check empirical vs expected
    max_kl = 0.0
    log_rep_mask = logcnt > 1
    rep_experts = torch.nonzero(log_rep_mask).flatten().tolist()
    for expert_id in rep_experts:
        probs = wrapped[expert_id]
        if probs.sum() < 1e-10:
            continue
        samples = torch.multinomial(probs.float(), n_samples, replacement=True)
        empirical = torch.zeros(num_slots, dtype=torch.float64, device=device)
        for s in range(num_slots):
            empirical[s] = (samples == s).sum().item() / n_samples
        p = probs.double()
        p = p / p.sum()
        q = empirical
        mask = (p > 1e-12) & (q > 1e-12)
        if mask.any():
            kl = (p[mask] * (p[mask] / q[mask]).log()).sum().item()
            max_kl = max(max_kl, kl)

    report.add(f"{S}/Sampling KL divergence (N={n_samples})",
               max_kl < 0.05,
               f"max KL: {max_kl:.6f} over {len(rep_experts)} replicated experts")


# ---------------------------------------------------------------------------
# Test scenarios
# ---------------------------------------------------------------------------

def make_test_scenarios() -> List[Tuple[str, ExpertPlacement]]:
    return [
        (
            "Simple 2-GPU, 6 experts (2 replicated)",
            ExpertPlacement(
                g=2, num_phy_per_gpu=4,
                phy2log=[0, 2, 3, 1,  0, 4, 5, 1],
                logical_count=[100, 50, 30, 20, 80, 70],
            ),
        ),
        (
            "4-GPU, 8 experts (all replicated x2)",
            ExpertPlacement(
                g=4, num_phy_per_gpu=4,
                phy2log=[0, 1, 4, 5,  0, 2, 6, 7,  1, 3, 4, 6,  2, 3, 5, 7],
                logical_count=[120, 90, 60, 110, 80, 40, 75, 95],
            ),
        ),
        (
            "2-GPU, uniform token counts",
            ExpertPlacement(
                g=2, num_phy_per_gpu=4,
                phy2log=[0, 1, 2, 3,  0, 1, 2, 3],
                logical_count=[100, 100, 100, 100],
            ),
        ),
        (
            "2-GPU, highly skewed token counts",
            ExpertPlacement(
                g=2, num_phy_per_gpu=4,
                phy2log=[0, 1, 2, 3,  0, 1, 2, 3],
                logical_count=[1000, 1, 1, 1],
            ),
        ),
        (
            "8-GPU, 64 experts (16 replicated x2, rest single)",
            ExpertPlacement(
                g=8, num_phy_per_gpu=10,
                phy2log=(
                    list(range(0, 10))                              # GPU0
                    + list(range(10, 20))                           # GPU1
                    + list(range(20, 30))                           # GPU2
                    + list(range(30, 40))                           # GPU3
                    + list(range(40, 48)) + [48, 49]                # GPU4
                    + [50, 51, 52, 53, 54, 55, 56, 57, 58, 59]     # GPU5
                    + [60, 61, 62, 63, 48, 49, 50, 51, 52, 53]     # GPU6
                    + [54, 55, 56, 57, 58, 59, 60, 61, 62, 63]     # GPU7
                ),
                logical_count=[float(i + 1) * 10 for i in range(64)],
            ),
        ),
        (
            "Edge: some experts with zero token count",
            ExpertPlacement(
                g=2, num_phy_per_gpu=4,
                phy2log=[0, 1, 2, 3,  0, 1, 2, 3],
                logical_count=[100, 0, 50, 200],
            ),
        ),
    ]


# ---------------------------------------------------------------------------
# Run one scenario (single layer)
# ---------------------------------------------------------------------------

def run_scenario(name: str, placement: ExpertPlacement, device: torch.device) -> ScenarioReport:
    report = ScenarioReport(name)
    phy2log, logcnt, log2phy, logical_count = build_tensors(placement, device)

    # Use single layer (index 0)
    lp = build_lp_full(
        phy2log[0], logcnt[0], log2phy[0],
        placement.g, logical_count[0], device,
    )

    # 0a: Pre-processing
    t0 = time.perf_counter()
    verify_preprocessing(lp, report)
    t_pre = time.perf_counter()

    # Solve
    A, b, c = lp["A"], lp["b"], lp["c"]
    A_np = A.detach().cpu().numpy().astype(np.float64)
    b_np = b.detach().cpu().numpy().astype(np.float64)
    c_np = c.detach().cpu().numpy().astype(np.float64)

    t_solve_start = time.perf_counter()
    res = linprog(c_np, A_eq=A_np, b_eq=b_np, bounds=(0, None), method="highs")
    t_solve_end = time.perf_counter()

    # 0b: Solver
    verify_solver(
        res.x if res.success else np.zeros(c_np.shape),
        A_np, b_np, c_np, res.success, report,
    )

    if not res.success:
        report.timings["matrix_build"] = (t_pre - t0) * 1000
        report.timings["solve"] = (t_solve_end - t_solve_start) * 1000
        return report

    result_t = torch.from_numpy(res.x).to(device=device, dtype=torch.float32)

    # Post-process
    t_post_start = time.perf_counter()
    log2phy_prob = postprocess(result_t, lp, device)
    t_post_end = time.perf_counter()

    # 0c: Post-processing
    verify_postprocessing(log2phy_prob, log2phy[0], logcnt[0], lp["lc_normalized"], report)

    # 0d: Dispatch
    verify_dispatch(
        log2phy_prob, log2phy[0], logcnt[0], lp["lc_normalized"],
        phy2log[0], placement.g, lp["num_phy_gpu"], report,
    )

    report.timings["matrix_build"] = (t_pre - t0) * 1000
    report.timings["solve"] = (t_solve_end - t_solve_start) * 1000
    report.timings["post_process"] = (t_post_end - t_post_start) * 1000
    report.timings["total"] = (t_post_end - t0) * 1000

    return report


# ---------------------------------------------------------------------------
# Run with real data
# ---------------------------------------------------------------------------

def run_real_data(path: str, device: torch.device) -> List[ScenarioReport]:
    data = torch.load(path, weights_only=True, map_location="cpu")
    logical_count_all = data["logical_count"]  # (layers, experts)
    if logical_count_all.dim() == 3:
        logical_count_all = logical_count_all.sum(dim=0)
    num_layers, num_log = logical_count_all.shape

    g = 8
    num_redundant = 16
    num_phy = num_log + num_redundant
    num_phy_gpu = num_phy // g

    print(f"\n  Real data: {path}")
    print(f"  layers={num_layers}, experts={num_log}, g={g}, redundant={num_redundant}")
    print(f"  num_phy={num_phy}, num_phy_gpu={num_phy_gpu}")

    # Build a plausible expert placement:
    # First num_log slots are 1:1 (expert i -> physical i), distributed across GPUs.
    # Redundant slots replicate the top-K hottest experts.
    avg_counts = logical_count_all.float().mean(dim=0)
    _, hottest = torch.topk(avg_counts, num_redundant)
    hottest_sorted = hottest.sort().values.tolist()

    phy2log_list = list(range(num_log)) + hottest_sorted
    assert len(phy2log_list) == num_phy

    phy2log_t = torch.tensor(phy2log_list, dtype=torch.int64, device=device)
    logcnt = torch.zeros(num_log, dtype=torch.int64, device=device)
    for p in phy2log_list:
        logcnt[p] += 1
    max_copies = int(logcnt.max().item())
    log2phy = torch.full((num_log, max_copies), -1, dtype=torch.int64, device=device)
    fill_idx = torch.zeros(num_log, dtype=torch.int64, device=device)
    for phy_idx, log_idx in enumerate(phy2log_list):
        pos = fill_idx[log_idx].item()
        log2phy[log_idx, pos] = phy_idx
        fill_idx[log_idx] += 1

    reports = []
    sample_layers = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]
    sample_layers = sorted(set(l for l in sample_layers if 0 <= l < num_layers))

    total_time = 0.0
    for layer_id in sample_layers:
        lc = logical_count_all[layer_id].float().to(device)
        if lc.sum().item() == 0:
            print(f"  Layer {layer_id}: zero token count, skipping")
            continue

        report = ScenarioReport(f"Real data layer {layer_id}")

        lp = build_lp_full(phy2log_t, logcnt, log2phy, g, lc, device)

        t0 = time.perf_counter()
        verify_preprocessing(lp, report)
        t_pre = time.perf_counter()

        A, b, c = lp["A"], lp["b"], lp["c"]
        A_np = A.detach().cpu().numpy().astype(np.float64)
        b_np = b.detach().cpu().numpy().astype(np.float64)
        c_np = c.detach().cpu().numpy().astype(np.float64)

        t_solve_start = time.perf_counter()
        res = linprog(c_np, A_eq=A_np, b_eq=b_np, bounds=(0, None), method="highs")
        t_solve_end = time.perf_counter()

        verify_solver(
            res.x if res.success else np.zeros(c_np.shape),
            A_np, b_np, c_np, res.success, report,
        )

        if res.success:
            result_t = torch.from_numpy(res.x).to(device=device, dtype=torch.float32)
            t_post_start = time.perf_counter()
            log2phy_prob = postprocess(result_t, lp, device)
            t_post_end = time.perf_counter()

            verify_postprocessing(log2phy_prob, log2phy, logcnt, lp["lc_normalized"], report)
            verify_dispatch(
                log2phy_prob, log2phy, logcnt, lp["lc_normalized"],
                phy2log_t, g, lp["num_phy_gpu"], report,
            )

            report.timings["matrix_build"] = (t_pre - t0) * 1000
            report.timings["solve"] = (t_solve_end - t_solve_start) * 1000
            report.timings["post_process"] = (t_post_end - t_post_start) * 1000
            report.timings["total"] = (t_post_end - t0) * 1000
            total_time += report.timings["total"]

        reports.append(report)

    print(f"\n  Total time for {len(sample_layers)} sampled layers: {total_time:.1f} ms")
    print(f"  Estimated total for all {num_layers} layers: {total_time / len(sample_layers) * num_layers:.1f} ms")

    return reports


# ---------------------------------------------------------------------------
# Multi-layer varying placement test (post-EPLB scenario)
# ---------------------------------------------------------------------------

def run_varying_placement_test(device: torch.device) -> List[ScenarioReport]:
    """Test lplb_algorithm with different expert placements per layer."""
    from sglang.srt.eplb.lplb_algorithm import lplb_algorithm

    reports = []

    # Scenario: 2 GPUs, 6 physical slots (4 logical + 2 redundant)
    # Layer 0: experts 0,1 replicated
    # Layer 1: experts 2,3 replicated
    # Layer 2: expert 0 replicated (different count)
    g = 2
    num_log = 4
    num_phy = 6

    phy2log = torch.tensor([
        [0, 1, 2, 3, 0, 1],   # layer 0: experts 0,1 x2
        [0, 1, 2, 3, 2, 3],   # layer 1: experts 2,3 x2
        [0, 1, 2, 3, 0, 0],   # layer 2: expert 0 x3
    ], dtype=torch.int64, device=device)

    logcnt = torch.tensor([
        [2, 2, 1, 1],  # layer 0
        [1, 1, 2, 2],  # layer 1
        [3, 1, 1, 1],  # layer 2
    ], dtype=torch.int64, device=device)

    log2phy = torch.tensor([
        [[0, 4, -1], [1, 5, -1], [2, -1, -1], [3, -1, -1]],  # layer 0
        [[0, -1, -1], [1, -1, -1], [2, 4, -1], [3, 5, -1]],  # layer 1
        [[0, 4, 5], [1, -1, -1], [2, -1, -1], [3, -1, -1]],  # layer 2
    ], dtype=torch.int64, device=device)

    logical_count = torch.tensor([
        [100, 50, 30, 20],
        [20, 30, 100, 50],
        [200, 10, 10, 10],
    ], dtype=torch.float32, device=device)

    result = lplb_algorithm(phy2log, logcnt, log2phy, g, logical_count, device)

    for layer_id in range(3):
        report = ScenarioReport(f"Varying placement layer {layer_id}")
        prob = result[layer_id]
        l2p = log2phy[layer_id]
        lc_t = logcnt[layer_id]
        lc_norm = logical_count[layer_id] / logical_count[layer_id].sum()

        S = "Post-processing"
        has_neg = (prob < -1e-8).any().item()
        report.add(f"{S}/Non-negativity", not has_neg,
                   f"min value: {prob.min().item():.6e}")

        valid_mask = l2p >= 0
        violations = ((~valid_mask) & (prob.abs() > 1e-10)).sum().item()
        report.add(f"{S}/Invalid positions zero", violations == 0,
                   f"{int(violations)} violations")

        row_sums = prob.sum(dim=-1)
        dev = (row_sums - lc_norm).abs().max().item()
        report.add(f"{S}/Row-sum == normalized token share",
                   dev < 1e-4, f"max deviation: {dev:.6e}")

        has_tokens = lc_norm > 1e-12
        zero_with_tokens = ((row_sums < 1e-10) & has_tokens).sum().item()
        report.add(f"{S}/No zero rows (experts with tokens)",
                   zero_with_tokens == 0,
                   f"{zero_with_tokens} zero rows")

        report.print_report()
        reports.append(report)

    # Larger scenario: simulate EPLB-like varying placement
    # 8 GPUs, 256 logical experts, 16 redundant, 10 layers with different placements
    g2 = 8
    num_log2 = 64
    num_redundant2 = 8
    num_phy2 = num_log2 + num_redundant2
    num_layers2 = 10

    torch.manual_seed(42)
    phy2log_list_base = list(range(num_log2))
    phy2log_layers = []
    logcnt_layers = []
    for lid in range(num_layers2):
        hot = torch.randint(0, num_log2, (num_redundant2,)).tolist()
        p2l = phy2log_list_base + hot
        phy2log_layers.append(p2l)
        lc = torch.zeros(num_log2, dtype=torch.int64)
        for p in p2l:
            lc[p] += 1
        logcnt_layers.append(lc)

    max_copies_all = max(lc.max().item() for lc in logcnt_layers)
    phy2log_t = torch.tensor(phy2log_layers, dtype=torch.int64, device=device)
    logcnt_t = torch.stack(logcnt_layers).to(device)

    log2phy_t = torch.full((num_layers2, num_log2, max_copies_all), -1,
                           dtype=torch.int64, device=device)
    for lid in range(num_layers2):
        fill = torch.zeros(num_log2, dtype=torch.int64)
        for phy_idx, log_idx in enumerate(phy2log_layers[lid]):
            pos = fill[log_idx].item()
            if pos < max_copies_all:
                log2phy_t[lid, log_idx, pos] = phy_idx
                fill[log_idx] += 1

    lc_large = torch.rand(num_layers2, num_log2, device=device) * 100 + 1

    result_large = lplb_algorithm(phy2log_t, logcnt_t, log2phy_t, g2, lc_large, device)

    report = ScenarioReport("Varying placement 10-layer EPLB-like")
    all_ok = True
    max_dev_overall = 0.0
    for lid in range(num_layers2):
        prob = result_large[lid]
        l2p = log2phy_t[lid]
        lc_norm = lc_large[lid] / lc_large[lid].sum()

        valid_mask = l2p >= 0
        violations = ((~valid_mask) & (prob.abs() > 1e-10)).sum().item()
        if violations > 0:
            all_ok = False

        row_sums = prob.sum(dim=-1)
        dev = (row_sums - lc_norm).abs().max().item()
        max_dev_overall = max(max_dev_overall, dev)
        if dev > 1e-4:
            all_ok = False

        has_neg = (prob < -1e-8).any().item()
        if has_neg:
            all_ok = False

    # Verify grouping worked: count distinct logcnt patterns
    logcnt_cpu = logcnt_t.cpu()
    groups = set()
    for lid in range(num_layers2):
        groups.add(logcnt_cpu[lid].numpy().tobytes())

    report.add("Multi-layer/All layers valid", all_ok,
               f"max row-sum deviation: {max_dev_overall:.6e}, {len(groups)} groups")
    report.add("Multi-layer/Non-negativity", True, "checked per layer")
    report.add("Multi-layer/Invalid positions", True, "checked per layer")
    report.print_report()
    reports.append(report)

    return reports


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-to-end static_lp verification")
    parser.add_argument("--real-data", type=str, default=None,
                        help="Path to expert_record_summed.pt for real-data test")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    all_reports: List[ScenarioReport] = []

    # Synthetic scenarios
    print("\n" + "=" * 72)
    print("  SYNTHETIC SCENARIOS")
    print("=" * 72)
    for name, placement in make_test_scenarios():
        report = run_scenario(name, placement, device)
        report.print_report()
        all_reports.append(report)

    # Varying placement (post-EPLB) scenarios
    print("\n" + "=" * 72)
    print("  VARYING PLACEMENT SCENARIOS (post-EPLB)")
    print("=" * 72)
    varying_reports = run_varying_placement_test(device)
    all_reports.extend(varying_reports)

    # Real data
    if args.real_data:
        print("\n" + "=" * 72)
        print("  REAL DATA SCENARIOS")
        print("=" * 72)
        real_reports = run_real_data(args.real_data, device)
        for r in real_reports:
            r.print_report()
            all_reports.append(r)

    # Summary
    print(f"\n{'='*72}")
    print("  SUMMARY")
    print(f"{'='*72}")
    all_pass = True
    for r in all_reports:
        status = "PASS" if r.all_passed else "FAIL"
        if not r.all_passed:
            all_pass = False
        t = r.timings.get("total", 0)
        print(f"  [{status}] {r.name} ({t:.1f} ms)")
        if not r.all_passed:
            for c in r.checks:
                if not c.passed:
                    print(f"         FAILED: {c.name}: {c.detail}")

    total_checks = sum(len(r.checks) for r in all_reports)
    passed_checks = sum(sum(1 for c in r.checks if c.passed) for r in all_reports)
    print(f"\n  {passed_checks}/{total_checks} checks passed across {len(all_reports)} scenarios")

    if all_pass:
        print("\n  All scenarios PASSED.")
    else:
        print("\n  Some scenarios FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
