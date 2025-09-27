import swyft
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialDropout1D(nn.Dropout2d):
    """Spatial dropout for 1D convs (drops entire channels)."""

    def forward(self, x):
        x = x.unsqueeze(2)  # (N, C, 1, L)
        x = super().forward(x)
        return x.squeeze(2)


class Network(swyft.SwyftModule):
    def __init__(self, input_length=250, dropout_rate=0.1):
        super().__init__()

        # --- Convolutional blocks (similar to Olmschenk et al.) ---
        conv_blocks = []
        in_channels = 1
        channels = [16, 32, 64]
        for i, out_channels in enumerate(channels):
            block = []
            block.append(nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1))
            block.append(nn.LeakyReLU())
            if i < len(channels) - 1:  # spatial dropout for convs except last
                block.append(SpatialDropout1D(dropout_rate))
            else:  # last conv before dense: standard dropout
                block.append(nn.Dropout(dropout_rate))
            if i < 2:  # first two conv blocks use pooling (like paper's first 6 convs)
                block.append(nn.MaxPool1d(2))
            if i not in [0, len(channels) - 1]:  # no BN on first or last conv block
                block.append(nn.BatchNorm1d(out_channels))
            conv_blocks.append(nn.Sequential(*block))
            in_channels = out_channels

        self.conv_layers = nn.Sequential(*conv_blocks)

        # figure out flatten size
        dummy = torch.zeros(1, 1, input_length)
        with torch.no_grad():
            out = self.conv_layers(dummy)
        flatten_size = out.shape[1] * out.shape[2]

        # --- Dense blocks ---
        self.fc1 = nn.Linear(flatten_size, 512)  # first dense block, no dropout/BN
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 16)

        self.dropout = nn.Dropout(dropout_rate)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(64)

        # --- Log-ratio estimator ---
        self.logratios = swyft.LogRatioEstimator_1dim(num_features=16, num_params=1, varnames="z")

    def forward(self, A, B):
        x = A["x"]  # (batch, 250)
        x = x.unsqueeze(1)  # (batch, 1, 250)

        # conv pipeline
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)

        # dense pipeline
        x = F.leaky_relu(self.fc1(features))
        x = F.leaky_relu(self.fc2(x))
        x = self.dropout(x)
        x = self.bn2(x)
        x = F.leaky_relu(self.fc3(x))
        x = self.dropout(x)
        x = self.bn3(x)
        embedding = F.leaky_relu(self.fc4(x))

        # log-ratio
        logratios = self.logratios(embedding, B["z"].unsqueeze(-1))
        return logratios
        return logratios
