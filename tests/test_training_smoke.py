import torch

from fedadaptops.models.simple_cnn import SimpleCNN


def test_simple_cnn_forward_shape():
    model = SimpleCNN(num_classes=10)
    x = torch.randn(4, 3, 32, 32)
    logits = model(x)

    assert logits.shape == (4, 10)
