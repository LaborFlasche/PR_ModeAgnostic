"""Neural network architectures for tabular data."""

import torch
import torch.nn as nn


class TabularMLP(nn.Module):
    """Multi-layer perceptron for tabular data.

    Builds a stack of Linear -> ReLU layers followed by a final Linear output layer.
    """

    def __init__(self, in_features: int, out_features: int, hidden_sizes: list[int]):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TabularTransformer(nn.Module):
    """Transformer encoder for tabular data.

    Each input feature is treated as a token: the scalar is projected to d_model
    dimensions, then processed by a TransformerEncoder stack. The output tokens
    are flattened and mapped to the desired output size.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
    ):
        super().__init__()
        self.in_features = in_features
        self.d_model = d_model

        self.input_proj = nn.Linear(1, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Linear(in_features * d_model, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = x.unsqueeze(-1)  # (batch, n_features, 1)
        x = self.input_proj(x)  # (batch, n_features, d_model)
        x = self.encoder(x)  # (batch, n_features, d_model)
        x = x.flatten(start_dim=1)  # (batch, n_features * d_model)
        return self.head(x)  # (batch, out_features)


class TabularCNN1D(nn.Module):
    """1-D convolutional network for tabular data.

    The feature vector is treated as a single-channel 1-D signal. A Conv1d layer
    extracts local patterns, followed by adaptive average pooling and a linear head.
    Works even when in_features < kernel_size thanks to padding.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_filters: int = 64,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.conv = nn.Conv1d(1, n_filters, kernel_size, padding=(kernel_size - 1) // 2)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(n_filters, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        x = x.unsqueeze(1)  # (batch, 1, n_features)
        x = self.conv(x)  # (batch, n_filters, n_features)
        x = self.relu(x)
        x = self.pool(x)  # (batch, n_filters, 1)
        x = x.squeeze(-1)  # (batch, n_filters)
        return self.head(x)  # (batch, out_features)
