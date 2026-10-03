"""Sequence models over the packets of one window (features/sequences.py).

Input: x (batch, SEQ_LEN, n_features) zero-padded, lengths (batch,) >= 1.
Output: one logit per window (attack vs normal). All three are small on purpose
(tens of thousands of parameters) so they can score windows live on a CPU.
"""
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _padding_mask(lengths, seq_len):
    """True where a position is padding."""
    return torch.arange(seq_len, device=lengths.device)[None, :] >= lengths[:, None]


class CNN(nn.Module):
    """1D convolutions over the packet axis, max-pooled over real packets (as in LUCID)."""

    def __init__(self, n_features, channels=64, kernel=3, dropout=0.2):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel, padding=kernel // 2), nn.ReLU(),
            nn.Conv1d(channels, channels, kernel, padding=kernel // 2), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(channels, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, lengths):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2)            # (B, L, C)
        h = h.masked_fill(_padding_mask(lengths, x.shape[1])[..., None], float("-inf"))
        return self.head(h.max(dim=1).values).squeeze(-1)


class LSTM(nn.Module):
    """One LSTM layer; the window is summarised by the state after its last real packet."""

    def __init__(self, n_features, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return self.head(h[-1]).squeeze(-1)


class Transformer(nn.Module):
    """Two self-attention layers with learned positions, mean-pooled over real packets."""

    def __init__(self, n_features, seq_len, d_model=64, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Linear(n_features, d_model)
        self.position = nn.Parameter(torch.zeros(1, seq_len, d_model))
        layer = nn.TransformerEncoderLayer(d_model, heads, dim_feedforward=2 * d_model,
                                           dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x, lengths):
        pad = _padding_mask(lengths, x.shape[1])
        h = self.encoder(self.embed(x) + self.position[:, :x.shape[1]], src_key_padding_mask=pad)
        h = h.masked_fill(pad[..., None], 0.0).sum(dim=1) / lengths[:, None]
        return self.head(h).squeeze(-1)


def build(name, n_features, seq_len):
    return {"CNN": lambda: CNN(n_features),
            "LSTM": lambda: LSTM(n_features),
            "Transformer": lambda: Transformer(n_features, seq_len)}[name]()
