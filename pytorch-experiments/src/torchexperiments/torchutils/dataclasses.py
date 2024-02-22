from dataclasses import dataclass
from collections.abc import Callable

import torch.nn as nn
import torch.optim as optim

from .constants import (
    ACTIVATION_TYPES,
    ARGPARSE_USE_DEFAULT_STR,
    LOSS_FN_TYPES,
    OPTIMIZER_TYPES,
)


def _get_key_with_value(dictionary, value):
    return list(dictionary.keys())[list(dictionary.values()).index(value)]


@dataclass(frozen=True)
class TrainingPerformance:
    training_time: float
    first_loss: float
    last_interval_loss: float
    last_loss: float

    def get_as_csv_string(self) -> str:
        return (
            f"{self.first_loss},{self.last_interval_loss},{self.last_loss},"
            + f"{self.training_time},"
        )


@dataclass(frozen=True)
class ValidationPerformance:
    validation_count: int
    correct_validation_count: int

    def get_as_csv_string(self) -> str:
        return f"{self.validation_count},{self.correct_validation_count},"


@dataclass(frozen=True)
class TrainingParameters:
    epochs: int
    learning_rate: float
    loss_restart: float
    loss_stop: float
    epoch_interval: int
    loss_fn_type: Callable
    optimizer: Callable

    def get_as_csv_string(self) -> str:

        return (
            f"{self.epochs},{self.learning_rate},{self.epoch_interval},"
            + f"{_get_key_with_value(LOSS_FN_TYPES, self.loss_fn_type)},"
            + f"{_get_key_with_value(OPTIMIZER_TYPES, self.optimizer)},"
        )


@dataclass(frozen=True)
class NetworkParameters:
    first_layer_size: int
    first_layer_activation: str
    second_layer_size: int
    second_layer_activation: str

    def get_first_layer_size(self, default: int) -> int:
        return default if self.first_layer_size == -1 else self.first_layer_size

    def has_second_layer(self) -> bool:
        return self.second_layer_size != 0

    def get_second_layer_size(self, default: int) -> int:
        if not self.has_second_layer():
            raise Exception(
                "Second layer size cannot be retrieved if no second layer should be used."
            )

        return default if self.second_layer_size == -1 else self.second_layer_size

    def get_first_layer_activation(self, default: Callable) -> Callable:
        return (
            default
            if self.first_layer_activation == ARGPARSE_USE_DEFAULT_STR
            else ACTIVATION_TYPES[self.first_layer_activation.lower()]
        )

    def get_second_layer_activation(self, default: Callable) -> Callable:
        if not self.has_second_layer():
            raise Exception(
                "Second layer activation cannot be retrieved if no second layer should be used."
            )

        return (
            default
            if self.second_layer_activation == ARGPARSE_USE_DEFAULT_STR
            else ACTIVATION_TYPES[self.second_layer_activation.lower()]
        )

    def get_as_csv_string(self) -> str:
        return (
            f"{self.first_layer_size}," + f"{self.first_layer_activation},"
            f"{self.second_layer_size}," + f"{self.second_layer_activation},"
        )


def optimizer_factory(
    network: nn.Module, training_params: TrainingParameters
) -> optim.Optimizer:
    optimizer = training_params.optimizer

    if optimizer == optim.SGD:
        optimizer = optimizer(
            network.parameters(), training_params.learning_rate, momentum=0.1
        )
    else:
        optimizer = optimizer(network.parameters(), training_params.learning_rate)

    return optimizer


def print_network_evaluation_as_human_readable(
    training_perf: TrainingPerformance, validation_perf: ValidationPerformance
):
    percentage_correct = (
        validation_perf.correct_validation_count
        / validation_perf.validation_count
        * 100
    )

    print("")
    print("")
    print("====================")
    print("Finished training...")
    print(
        f"{validation_perf.correct_validation_count}/{validation_perf.validation_count} correct "
        + f"({percentage_correct:.1f}%) in {training_perf.training_time}s"
    )


def print_network_evaluation_as_csv(
    training_params: TrainingParameters,
    network_params: NetworkParameters,
    training_perf: TrainingPerformance,
    validation_perf: ValidationPerformance,
):
    print(
        training_params.get_as_csv_string()
        + network_params.get_as_csv_string()
        + training_perf.get_as_csv_string()
        + validation_perf.get_as_csv_string()
    )


def print_network_evaluation(
    output_format: str,
    training_params: TrainingParameters,
    network_params: NetworkParameters,
    training_perf: TrainingPerformance,
    validation_perf: ValidationPerformance,
):
    if output_format == "human":
        print_network_evaluation_as_human_readable(training_perf, validation_perf)
    else:
        print_network_evaluation_as_csv(
            training_params, network_params, training_perf, validation_perf
        )
