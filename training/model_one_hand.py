"""One-handed landmark classifier (ASL/ISL).

A deliberately tiny MLP: 63 floats in, 26 logits out, ~12k parameters. For a
well-bounded task with good features, small models with clean data win.
"""

import torch.nn as nn


class OneHandClassifier(nn.Module):
    def __init__(self, num_classes: int = 26):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)
