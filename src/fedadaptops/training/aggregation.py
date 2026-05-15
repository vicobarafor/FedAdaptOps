from __future__ import annotations

import torch

from fedadaptops.clients.client import ClientTrainResult


def sample_weighted_average(results: list[ClientTrainResult]) -> dict[str, torch.Tensor]:
    if not results:
        raise ValueError("Cannot aggregate an empty list of client results.")
    total_samples = sum(result.num_samples for result in results)
    if total_samples <= 0:
        raise ValueError("Total sample count must be positive.")
    aggregated: dict[str, torch.Tensor] = {}
    for key in results[0].state_dict:
        reference = results[0].state_dict[key]
        if not torch.is_floating_point(reference):
            aggregated[key] = reference.clone()
            continue
        weighted_sum = torch.zeros_like(reference, dtype=torch.float32)
        for result in results:
            tensor = result.state_dict[key].to(dtype=torch.float32)
            weighted_sum += tensor * (result.num_samples / total_samples)
        aggregated[key] = weighted_sum.to(dtype=reference.dtype)
    return aggregated
