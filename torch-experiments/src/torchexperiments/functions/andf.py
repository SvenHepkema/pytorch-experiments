import argparse
import random
from typing import Any

import torch
import torch.nn as nn

from torchexperiments.torchutils.constants import DEVICE
from torchexperiments.functions.experiment import Experiment


class _ANDNet(nn.Module):
    def __init__(self):
        super(_ANDNet, self).__init__()
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


def _generate_and_network() -> nn.Module:
    return _ANDNet().to(DEVICE)


def _generate_and_data(n: int):
    data = []

    for _ in range(n):
        data.append((random.uniform(0, 1), random.uniform(0, 1)))

    return data


def _convert_and_data_to_tensor(data) -> torch.Tensor:
    return torch.FloatTensor(data).to(DEVICE)


def _evaluate_and(gates: tuple):
    a = gates[0]
    b = gates[1]

    if a > 0.5 and b > 0.5:
        return 1.0
    else:
        return 0.0


def _evaluate_if_labels_are_equal(label: Any, output: torch.Tensor) -> bool:
    return int(round(label)) == int(round(output.item()))


def get_and_experiment(args: argparse.Namespace) -> Experiment:
    return Experiment(
        args=args,
        network_generator=_generate_and_network,
        function_evaluator=_evaluate_and,
        data_generator=_generate_and_data,
        data_to_tensor_converter=_convert_and_data_to_tensor,
        label_to_tensor_converter=_convert_and_data_to_tensor,
        label_equalness_evaluator=_evaluate_if_labels_are_equal,
    )
