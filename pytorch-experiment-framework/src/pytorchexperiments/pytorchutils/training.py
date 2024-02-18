import logging
import time
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataclasses import (
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

        if epoch % 100 == 0:
            logging.info(f"epoch {epoch} loss: {running_loss}")

            if epoch == 0:
                first_loss = running_loss

            if epoch == 100 and training_params.loss_stop != 0.0:
                if first_loss * (1 - training_params.loss_stop) < running_loss:
                    return None

    end_time = time.time()

    return TrainingPerformance(end_time - start_time, first_loss, running_loss)


def train_network(
    training_data: DataLoader,
    training_params: TrainingParameters,
    network_generator: Callable[[], nn.Module],
) -> tuple[TrainingPerformance, nn.Module]:
    training_perf = None
    network = network_generator()

    while training_perf is None:
        training_perf = train_network_with_stop(training_data, network, training_params)

        if training_perf is None:
            logging.info("Training the network failed, restarting")
            network = network_generator()
        else:
            break

    return training_perf, network


def evaluate_network(
    validation_data: torch.Tensor, network: nn.Module, evaluator: Callable
) -> ValidationPerformance:
    """Returns the number of correctly predicted labels for each record in validation data."""

    correct_count = 0
    output = network(validation_data)
    results = list(zip(validation_data, output))
    for validation_data, output in results:
        correct = int(round(evaluator(validation_data))) == int(round(output.item()))
        correct_count += int(correct)
        logging.debug(f"{validation_data} \t=>\t {output.item()} | {correct}")

    return ValidationPerformance(len(results), correct_count)
