import pytest

torch = pytest.importorskip("torch")
from pucb.util import uniform_sampling


def test_uniform_sampling_bounds_shape():
    bounds = torch.tensor([[0.0, -1.0], [1.0, 1.0]])
    n = 5
    samples = uniform_sampling(n, bounds)
    assert samples.shape == (n, bounds.shape[1])
    assert torch.all(samples >= bounds[0])
    assert torch.all(samples <= bounds[1])
