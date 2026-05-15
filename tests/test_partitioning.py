import numpy as np

from fedadaptops.data.partitioning import dirichlet_partition


def test_dirichlet_partition_covers_all_indices_once():
    labels = np.array([0] * 50 + [1] * 50)
    partitions = dirichlet_partition(
        labels=labels,
        num_clients=5,
        alpha=0.5,
        seed=123,
        min_samples_per_client=1,
    )

    all_indices = [idx for client_indices in partitions.values() for idx in client_indices]

    assert len(partitions) == 5
    assert sorted(all_indices) == list(range(100))
    assert len(set(all_indices)) == 100


def test_dirichlet_partition_is_reproducible():
    labels = np.array([0] * 50 + [1] * 50)

    p1 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=123)
    p2 = dirichlet_partition(labels, num_clients=5, alpha=0.5, seed=123)

    assert p1 == p2
