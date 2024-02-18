import argparse
from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchexperiments.utils.argparseutils import (
    get_training_parameters_from_args,
)
from torchexperiments.torchutils.training import evaluate_network, train_network
from torchexperiments.torchutils.dataclasses import (
    TrainingParameters,
    TrainingPerformance,
    ValidationPerformance,
)


@dataclass(frozen=True)
class Experiment:
    args: argparse.Namespace
    network_generator: Callable[[], nn.Module]
    function_evaluator: Callable[..., Any]
    data_generator: Callable[[int], Any]
    data_to_tensor_converter: Callable[..., torch.Tensor]
    label_to_tensor_converter: Callable[..., torch.Tensor]

    def _generate_dataloader(self) -> DataLoader:
        data = self.data_generator(self.args.training_size)
        labels = [[self.function_evaluator(value)] for value in data]

        data_tensor = self.data_to_tensor_converter(data)
        labels_tensor = self.label_to_tensor_converter(labels)

        return DataLoader(
            list(zip(data_tensor, labels_tensor)),
            shuffle=True,
            batch_size=self.args.batch_size,
        )

    def run(
        self,
    ) -> tuple[TrainingParameters, TrainingPerformance, ValidationPerformance]:
        training_params = get_training_parameters_from_args(self.args)
        training_dataloader = self._generate_dataloader()
        training_perf, network = train_network(
            training_dataloader, training_params, self.network_generator
        )

        validation_data_tensor = self.data_to_tensor_converter(
            self.data_generator(self.args.validation_size)
        )
        validation_perf = evaluate_network(
            validation_data_tensor, network, self.function_evaluator
        )

        return training_params, training_perf, validation_perf
