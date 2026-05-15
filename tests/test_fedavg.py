import torch

from fedadaptops.clients.client import ClientTrainResult
from fedadaptops.training.aggregation import sample_weighted_average
from fedadaptops.training.fedavg import ClientSampler


def test_sample_weighted_average_uses_client_sample_counts():
    result_a = ClientTrainResult(0, 1, 0.0, 0.0, 1, {"w": torch.tensor([1.0])})
    result_b = ClientTrainResult(1, 3, 0.0, 0.0, 1, {"w": torch.tensor([3.0])})
    aggregated = sample_weighted_average([result_a, result_b])
    assert torch.allclose(aggregated["w"], torch.tensor([2.5]))


def test_client_sampler_is_reproducible():
    sampler_a = ClientSampler(
        num_clients=10, clients_per_round=4, dropout_probability=0.25, seed=123
    )
    sampler_b = ClientSampler(
        num_clients=10, clients_per_round=4, dropout_probability=0.25, seed=123
    )
    assert sampler_a.sample(round_id=1) == sampler_b.sample(round_id=1)


def test_client_sampler_guarantees_at_least_one_active_client():
    sampler = ClientSampler(num_clients=5, clients_per_round=3, dropout_probability=0.999, seed=123)
    assert len(sampler.sample(round_id=1).active_client_ids) >= 1
