from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientSelection:
    round_id: int
    selected_client_ids: list[int]
    active_client_ids: list[int]
    dropped_client_ids: list[int]


class ClientSampler:
    def __init__(
        self, *, num_clients: int, clients_per_round: int, dropout_probability: float, seed: int
    ):
        if clients_per_round > num_clients:
            raise ValueError("clients_per_round cannot exceed num_clients.")
        self.num_clients = num_clients
        self.clients_per_round = clients_per_round
        self.dropout_probability = dropout_probability
        self.rng = random.Random(seed)

    def sample(self, round_id: int) -> ClientSelection:
        selected = self.rng.sample(range(self.num_clients), self.clients_per_round)
        active: list[int] = []
        dropped: list[int] = []
        for client_id in selected:
            if self.rng.random() < self.dropout_probability:
                dropped.append(client_id)
            else:
                active.append(client_id)
        if not active:
            rescued = selected[0]
            active.append(rescued)
            dropped = [client_id for client_id in dropped if client_id != rescued]
        return ClientSelection(round_id, selected, active, dropped)
