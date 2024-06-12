import pytest
import torch

from avr.model.bottleneck import BottleneckFactory

BATCH_SIZE = 2
NUM_CHANNELS = 8
FEATURE_DIM = 32


@pytest.fixture
def x() -> torch.Tensor:
    return torch.rand(BATCH_SIZE, NUM_CHANNELS, FEATURE_DIM)


@pytest.mark.parametrize(
    "method,ratio",
    [
        ("maxpool", 0.125),
        ("maxpool", 0.25),
        ("maxpool", 0.5),
        ("maxpool", 1.0),
        ("avgpool", 0.125),
        ("avgpool", 0.25),
        ("avgpool", 0.5),
        ("avgpool", 1.0),
        ("linear", 0.125),
        ("linear", 0.25),
        ("linear", 0.5),
        ("linear", 1.0),
        ("conv", 0.25),
        ("conv", 0.5),
        ("conv", 1.0),
    ],
)
def test_bottleneck(x: torch.Tensor, method: str, ratio: float):
    model = BottleneckFactory.create(method=method, ratio=ratio, input_dim=FEATURE_DIM)
    y = model(x)
    assert y.shape == (BATCH_SIZE, NUM_CHANNELS, FEATURE_DIM * ratio)
