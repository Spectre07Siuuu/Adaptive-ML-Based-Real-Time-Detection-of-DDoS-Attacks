import pytest
import torch

from ddos.features.sequences import PACKET_FEATURES, SEQ_LEN
from ddos.training.deep_models import build

F = len(PACKET_FEATURES)


@pytest.mark.parametrize("name", ["CNN", "LSTM", "Transformer"])
def test_one_logit_per_window(name):
    torch.manual_seed(0)
    model = build(name, F, SEQ_LEN).eval()
    x = torch.rand(4, SEQ_LEN, F)
    lengths = torch.tensor([1, 5, SEQ_LEN, 10])
    x[torch.arange(SEQ_LEN)[None, :] >= lengths[:, None]] = 0     # zero padding, as stored
    out = model(x, lengths)
    assert out.shape == (4,) and torch.isfinite(out).all()


@pytest.mark.parametrize("name", ["LSTM", "Transformer"])
def test_padding_content_is_ignored(name):
    torch.manual_seed(0)
    model = build(name, F, SEQ_LEN).eval()
    x = torch.rand(3, SEQ_LEN, F)
    lengths = torch.tensor([2, 7, SEQ_LEN])
    noisy = x.clone()
    noisy[torch.arange(SEQ_LEN)[None, :] >= lengths[:, None]] = 5.0
    with torch.no_grad():
        assert torch.allclose(model(x, lengths), model(noisy, lengths), atol=1e-5)
