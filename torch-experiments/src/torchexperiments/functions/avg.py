import argparse
import random
from typing import Any

import torch
import torch.nn as nn

from torchexperiments.torchutils.constants import DEVICE
from torchexperiments.functions.experiment import Experiment

_LENGTH_INPUTS = 3
_DECIMAL_PREDICTION_PRECISION = 2


class _AVGNet(nn.Module):
    def __init__(self):
        super(_AVGNet, self).__init__()
        self.fc1 = nn.Linear(_LENGTH_INPUTS, 60)
        self.fc2 = nn.Linear(60, 1)
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x


def _generate_avg_network() -> nn.Module:
    return _AVGNet().to(DEVICE)


def _generate_avg_data(n: int):
    data = []

    for _ in range(n):
        inputs = []
        for _ in range(_LENGTH_INPUTS):
            inputs.append(random.uniform(0, 1))
        data.append(inputs)

    return data


def _convert_avg_data_to_tensor(data) -> torch.Tensor:
    return torch.FloatTensor(data).to(DEVICE)


def _evaluate_avg(values: list):
    if isinstance(values, torch.Tensor):
        values = values.tolist()

    return sum(values) / len(values)


def _evaluate_if_labels_are_equal(label: Any, output: torch.Tensor) -> bool:
    return round(label, _DECIMAL_PREDICTION_PRECISION) == round(
        output.item(), _DECIMAL_PREDICTION_PRECISION
    )


def get_avg_experiment(args: argparse.Namespace) -> Experiment:
    return Experiment(
        args=args,
        network_generator=_generate_avg_network,
        function_evaluator=_evaluate_avg,
        data_generator=_generate_avg_data,
        data_to_tensor_converter=_convert_avg_data_to_tensor,
        label_to_tensor_converter=_convert_avg_data_to_tensor,
        label_equalness_evaluator=_evaluate_if_labels_are_equal,
    )
