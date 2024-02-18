import argparse
import random

import torch
import torch.nn as nn

from pytorchexperiments.torchutils.constants import DEVICE
from pytorchexperiments.functions.experiment import Experiment


class _XORNet(nn.Module):
    def __init__(self):
        super(_XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 60)
        self.fc2 = nn.Linear(60, 1)
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x


def _generate_xor_network() -> nn.Module:
    return _XORNet().to(DEVICE)


def _generate_xor_data(n: int):
    data = []

    for _ in range(n):
        data.append((random.uniform(0, 1), random.uniform(0, 1)))

    return data


def _convert_xor_data_to_tensor(data) -> torch.Tensor:
    return torch.FloatTensor(data).to(DEVICE)


def _evaluate_xor(gates: tuple):
    a = gates[0]
    b = gates[1]

    if (a < 0.5 and b > 0.5) or (a > 0.5 and b < 0.5):
        return 1.0
    else:
        return 0.0


def get_xor_experiment(args: argparse.Namespace) -> Experiment:
    return Experiment(
        args=args,
        network_generator=_generate_xor_network,
        function_evaluator=_evaluate_xor,
        data_generator=_generate_xor_data,
        data_to_tensor_converter=_convert_xor_data_to_tensor,
        label_to_tensor_converter=_convert_xor_data_to_tensor,
    )
