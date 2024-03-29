import logging
import time
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataclasses import (
    NetworkParameters,
    TrainingParameters,
    TrainingPerformance,
    ValidationPerformance,
    optimizer_factory,
)


def train_network_with_stop(
    dataloader: DataLoader, network: nn.Module, training_params: TrainingParameters
) -> TrainingPerformance | None:
    loss_fn = training_params.loss_fn_type()
    optimizer = optimizer_factory(network, training_params)

    start_time = time.time()

    first_loss = 0.0
    last_interval_loss = 0.0
    running_loss = 0.0
    for epoch in range(training_params.epochs):
        running_loss = 0.0

        for data, labels in dataloader:
            optimizer.zero_grad()
            output = network(data)
            loss = loss_fn(output, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if epoch == 1:
            first_loss = running_loss

        if epoch % training_params.epoch_interval == 0:
            logging.info(f"epoch {epoch} loss: {running_loss}")

            if epoch == training_params.epoch_interval:
                loss_restart_threshold = first_loss * (1 - training_params.loss_restart)
                if (
                    training_params.loss_restart != 0.0
                    and loss_restart_threshold < running_loss
                ):
                    return None
            elif epoch > training_params.epoch_interval:
                loss_stop_threshold = last_interval_loss * (
                    1 - training_params.loss_stop
                )
                if (
                    training_params.loss_stop != 0.0
                    and loss_stop_threshold < running_loss
                ):
                    return TrainingPerformance(
                        time.time() - start_time,
                        first_loss,
                        last_interval_loss,
                        running_loss,
                    )
            last_interval_loss = running_loss

    return TrainingPerformance(
        time.time() - start_time, first_loss, last_interval_loss, running_loss
    )


def train_network(
    training_data: DataLoader,
    training_params: TrainingParameters,
    network_params: NetworkParameters,
    network_generator: Callable[[NetworkParameters], nn.Module],
) -> tuple[TrainingPerformance, nn.Module]:
    training_perf = None
    network = network_generator(network_params)

    while training_perf is None:
        training_perf = train_network_with_stop(training_data, network, training_params)

        if training_perf is None:
            logging.info("Training the network failed, restarting")
            network = network_generator(network_params)
        else:
            break

    return training_perf, network


def evaluate_network(
    validation_data: torch.Tensor,
    network: nn.Module,
    data_evaluator: Callable,
    label_equalness_evaluator: Callable[[Any, torch.Tensor], bool],
) -> ValidationPerformance:
    """Returns the number of correctly predicted labels for each record in validation data."""

    correct_count = 0
    output = network(validation_data)
    results = list(zip(validation_data, output))
    for validation_data, output in results:
        evaluation = data_evaluator(validation_data)
        correct = label_equalness_evaluator(evaluation, output)
        correct_count += int(correct)
        logging.debug(
            f"{validation_data} \t=>\t {output.item()} (should be: {evaluation} | {correct}"
        )

    return ValidationPerformance(len(results), correct_count)
