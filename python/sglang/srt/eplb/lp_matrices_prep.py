import time
import torch
from dataclasses import dataclass
from typing import List, Optional
from sglang.srt.server_args import ServerArgs


@dataclass
class TokenDispatchMetadata:
    A: List[torch.Tensor]
    B1: List[torch.Tensor]
    B2: List[torch.Tensor]
    c: List[torch.Tensor]
    log_single_expert_array: List[torch.Tensor]
    phy_single_expert_array: List[torch.Tensor]
    log_replicated_expert_array: List[torch.Tensor]
    phy_replicated_expert_array: List[torch.Tensor]

    @staticmethod
    def init(phy2log: torch.Tensor, log2phy: torch.Tensor, g: int):
        assert (
            phy2log.shape[0] == log2phy.shape[0]
        ), "phy2log, log2phy must have the same number of layers"
        assert (
            phy2log.shape[1] % g == 0
        ), "Number of physical experts must be divisible by number of GPUs"
        num_layers = phy2log.shape[0]
        num_logical = log2phy.shape[1]
        logcnt = torch.zeros(
            (num_layers, num_logical), dtype=torch.int, device=phy2log.device
        )
        for layer_id in range(num_layers):
            bincount = torch.bincount(phy2log[layer_id], minlength=num_logical)
            logcnt[layer_id] = bincount

        A = []
        B1 = []
        B2 = []
        c = []
        log_single_expert_array = []
        phy_single_expert_array = []
        log_replicated_expert_array = []
        phy_replicated_expert_array = []
        for i in range(num_layers):
            (
                A_i,
                B1_i,
                B2_i,
                c_i,
                log_single_expert_array_i,
                phy_single_expert_array_i,
                log_replicated_expert_array_i,
                phy_replicated_expert_array_i,
            ) = TokenDispatchMetadata.init_single_layer(
                phy2log[i], logcnt[i], log2phy[i], g
            )
            A.append(A_i)
            B1.append(B1_i)
            B2.append(B2_i)
            c.append(c_i)
            log_single_expert_array.append(log_single_expert_array_i)
            phy_single_expert_array.append(phy_single_expert_array_i)
            log_replicated_expert_array.append(log_replicated_expert_array_i)
            phy_replicated_expert_array.append(phy_replicated_expert_array_i)
        # Convert lists of tensors to tensors (stack along new dimension)
        # print(f"A: {[x.shape for x in A]}")
        # A = torch.stack(A)
        # B1 = torch.stack(B1)
        # B2 = torch.stack(B2)
        # c = torch.stack(c)
        # single_expert_array = torch.stack(single_expert_array)
        # log_replicated_expert_array = torch.stack(log_replicated_expert_array)
        # phy_replicated_expert_array = torch.stack(phy_replicated_expert_array)

        return TokenDispatchMetadata(
            A,
            B1,
            B2,
            c,
            log_single_expert_array,
            phy_single_expert_array,
            log_replicated_expert_array,
            phy_replicated_expert_array,
        )

    @staticmethod
    def init_single_layer(
        layer_phy2log: torch.Tensor,
        layer_logcnt: torch.Tensor,
        layer_log2phy: torch.Tensor,
        g: int,
    ):
        assert layer_phy2log.device == layer_logcnt.device, (
            "layer_phy2log and layer_logcnt must be on the same device, "
            f"{layer_phy2log.device=} != {layer_logcnt.device=}"
        )
        device = layer_phy2log.device
        num_phy: int = layer_phy2log.shape[0]
        num_phy_gpu: int = num_phy // g

        log_single_expert_array: torch.Tensor = torch.nonzero(
            layer_logcnt == 1
        ).flatten()
        phy_single_expert_array: torch.Tensor = layer_log2phy[log_single_expert_array, 0]
        log_replicated_expert_array: torch.Tensor = torch.nonzero(
            layer_logcnt > 1
        ).flatten()
        phy_replicated_expert_array: torch.Tensor = torch.nonzero(
            layer_logcnt[layer_phy2log] > 1
        ).flatten()

        single_expert_count: int = len(log_single_expert_array)
        log_replicated_expert_count: int = len(log_replicated_expert_array)
        phy_replicated_expert_count: int = len(phy_replicated_expert_array)

        B = torch.zeros((g, num_phy), dtype=torch.float32, device=device)
        for i in range(g):
            B[i, i * num_phy_gpu : (i + 1) * num_phy_gpu] = 1
        B1 = B[:, phy_single_expert_array]
        B2 = B[:, phy_replicated_expert_array]

        # Create C matrix using torch operations
        C = torch.zeros(
            (log_replicated_expert_count, phy_replicated_expert_count),
            dtype=torch.float32,
            device=device,
        )
        phy2log_rep = layer_phy2log[phy_replicated_expert_array]
        for i in range(log_replicated_expert_count):
            C[i, phy2log_rep == log_replicated_expert_array[i]] = 1.0

        # Construct matrix A = [[C, 0, 0, 1000], [B2, I, -1, 1000]]
        zeros_top = torch.zeros(
            (log_replicated_expert_count, g), dtype=torch.float32, device=device
        )
        zeros_top_col = torch.zeros(
            (log_replicated_expert_count, 1), dtype=torch.float32, device=device
        )

        I_matrix = torch.eye(g, dtype=torch.float32, device=device)
        neg_ones_col = torch.full((g, 1), -1.0, dtype=torch.float32, device=device)

        # Construct the matrix using torch block operations
        A_top = torch.hstack([C, zeros_top, zeros_top_col])
        A_bottom = torch.hstack([B2, I_matrix, neg_ones_col])
        A = torch.vstack([A_top, A_bottom])

        c = torch.zeros(A.shape[1] + 1, dtype=torch.float32, device=device)
        c[-2] = 1.0
        c[-1] = 1000.0

        return (
            A,
            B1,
            B2,
            c,
            log_single_expert_array,
            phy_single_expert_array,
            log_replicated_expert_array,
            phy_replicated_expert_array,
        )


_global_token_dispatch_metadata: Optional[TokenDispatchMetadata] = None


def get_global_token_dispatch_metadata():
    return _global_token_dispatch_metadata


def set_global_token_dispatch_metadata(value):
    global _global_token_dispatch_metadata

    _global_token_dispatch_metadata = value


def test():
    torch.manual_seed(42)
    device = "cuda"
    # Convert numpy arrays to torch tensors
    phy2log = torch.cat([torch.arange(256), torch.arange(32)]).repeat(58, 1).to(device)
    logcnt = (
        torch.cat([torch.full((32,), 2), torch.ones(224, dtype=torch.long)])
        .repeat(58, 1)
        .to(device)
    )
    log2phy = (
        torch.cat(
            [
                torch.stack([torch.arange(32), torch.arange(32) + 256], dim=1).to(
                    device
                ),
                torch.stack(
                    [torch.arange(32, 256), torch.full((224,), -1)], dim=1
                ).to(device),
            ],
            dim=0,
        )
        .repeat(58, 1, 1)
        .to(device)
    )

    g = 8

    start_time = time.time()
    token_dispatch_metadata = TokenDispatchMetadata.init(phy2log, log2phy, g)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print("TokenDispatchMetadata attributes:")
    for attr in [
        "A",
        "B1",
        "B2",
        "single_expert_array",
        "log_replicated_expert_array",
        "phy_replicated_expert_array",
    ]:
        tensor = getattr(token_dispatch_metadata, attr)
        shape = tensor.shape
        device = getattr(tensor, "device", None)
        print(f"{attr}: shape={shape}, device={device}")


if __name__ == "__main__":
    test()
