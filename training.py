from collections.abc import Callable
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda")

CRITERION_TYPES = {
    "crossentropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "nll": nn.NLLLoss,
    "bce": nn.BCELoss
}

OPTIMIZER_TYPES = {
    "sgd": optim.SGD,
    "adam": optim.Adam
}


@dataclass
class TrainingParameters:
    epochs: int
    learning_rate: float
    criterion_type: Callable
    optimizer: Callable

    def __init__(self, args):
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.criterion_type = CRITERION_TYPES[args.criterion.lower()]
        self.optimizer = OPTIMIZER_TYPES[args.optimizer.lower()]


def optimizer_factory(network, training_params: TrainingParameters):
    optimizer = training_params.optimizer

    if optimizer == optim.SGD:
        optimizer = optimizer(network.parameters(), training_params.learning_rate, momentum=0.1)
    else:
        optimizer = optimizer(network.parameters(), training_params.learning_rate)

    return optimizer


def train_network(dataloader: DataLoader, network: nn.Module, training_params: TrainingParameters) -> nn.Module:
    criterion = training_params.criterion_type()
    optimizer = optimizer_factory(network, training_params)

    for epoch in range(training_params.epochs):
        running_loss = 0.0

        for data, labels in dataloader:
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, labels)
            loss.backward()        
            optimizer.step()
            running_loss += loss.item()
            
        if epoch % 100 == 0:
            logging.info(f"epoch {epoch} loss: {running_loss}")

    return network 


def show_result(input: torch.Tensor, network: nn.Module, evaluator: Callable):
    print("")
    print("")
    print("====================")
    print("Finished training...")

    correct_count = 0
    output = network(input)
    results = list(zip(input, output))
    for input, output in results:
        correct = int(round(evaluator(input))) == int(round(output.item()))
        correct_count += int(correct)
        logging.debug(f"{input} \t=>\t {output.item()} | {correct}")

    print(f"{correct_count}/{len(results)} correct ({(correct_count/len(results)*100):.1f}%)")
