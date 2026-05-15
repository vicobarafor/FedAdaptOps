import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from fedadaptops.clients.client import FederatedClient


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


def test_federated_client_returns_state_dict_and_metrics():
    x = torch.randn(8, 2)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    model = TinyModel()
    initial_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    client = FederatedClient(client_id=7, train_loader=loader)
    result = client.train(
        model=model,
        global_state_dict=initial_state,
        device=torch.device("cpu"),
        local_epochs=1,
        lr=0.01,
        weight_decay=0.0,
        max_batches_per_epoch=None,
    )
    assert result.client_id == 7
    assert result.num_samples == 8
    assert result.local_epochs == 1
    assert "linear.weight" in result.state_dict
    assert isinstance(result.train_loss, float)
    assert isinstance(result.train_accuracy, float)
