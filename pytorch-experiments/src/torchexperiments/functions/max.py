import argparse
import random
from typing import Any

import torch
import torch.nn as nn

from torchexperiments.torchutils.constants import DEVICE
from torchexperiments.functions.experiment import Experiment

_LENGTH_INPUTS = 10
_DECIMAL_PREDICTION_PRECISION = 2
_MIDDLE_LAYER_SIZE = 300


class _MAXNet(nn.Module):
    def __init__(self):
        super(_MAXNet, self).__init__()
        self.fc1 = nn.Linear(_LENGTH_INPUTS, _MIDDLE_LAYER_SIZE)
        self.fc2 = nn.Linear(_MIDDLE_LAYER_SIZE, _MIDDLE_LAYER_SIZE)
        self.fc3 = nn.Linear(_MIDDLE_LAYER_SIZE, 1)
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
        self.rl3 = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)
        x = self.fc3(x)
        x = self.rl3(x)

        return x


def _generate_max_network() -> nn.Module:
    return _MAXNet().to(DEVICE)


def _generate_max_data(n: int):
    data = []

    for _ in range(n):
        inputs = []
        for _ in range(_LENGTH_INPUTS):
            inputs.append(random.uniform(0, 1))
        data.append(inputs)

    return data


def _convert_max_data_to_tensor(data) -> torch.Tensor:
    return torch.FloatTensor(data).to(DEVICE)


def _evaluate_max(values: list):
    if isinstance(values, torch.Tensor):
        values = values.tolist()

    return max(values)


def _evaluate_if_labels_are_equal(label: Any, output: torch.Tensor) -> bool:
    return round(label, _DECIMAL_PREDICTION_PRECISION) == round(
        output.item(), _DECIMAL_PREDICTION_PRECISION
    )


def get_max_experiment(args: argparse.Namespace) -> Experiment:
    return Experiment(
        args=args,
        network_generator=_generate_max_network,
        function_evaluator=_evaluate_max,
        data_generator=_generate_max_data,
        data_to_tensor_converter=_convert_max_data_to_tensor,
        label_to_tensor_converter=_convert_max_data_to_tensor,
        label_equalness_evaluator=_evaluate_if_labels_are_equal,
    )
