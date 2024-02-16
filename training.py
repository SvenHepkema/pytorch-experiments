from collections.abc import Callable
from dataclasses import dataclass
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

LOSS_FN_TYPES = {
    "crossentropy": nn.CrossEntropyLoss,
    "mse": nn.MSELoss,
    "l1": nn.L1Loss,
    "nll": nn.NLLLoss,
    "bce": nn.BCELoss
}

OPTIMIZER_TYPES = {
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "adam": optim.Adam,
    "adamw": optim.AdamW,
    "adamsparse": optim.SparseAdam,
    "adamax": optim.Adamax,
    "asgd": optim.ASGD,
    "sgd": optim.SGD,
    # "lbfgs": optim.LBFGS,  FIX: not usable until closure parameter is added
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
}


@dataclass
class TrainingParameters:
    epochs: int
    learning_rate: float
    loss_fn_type: Callable
    optimizer: Callable

    def __init__(self, args):
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.loss_fn_type = LOSS_FN_TYPES[args.loss_fn.lower()]
        self.optimizer = OPTIMIZER_TYPES[args.optimizer.lower()]


def add_training_params_to_parser(parser):
    parser.add_argument('-ep', '--epochs', type=int, default=5000,
                        help="number of epochs to run")
    parser.add_argument('-bs', '--batch-size', type=int, default=100,
                        help="number of samples per batch")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help="learning rate of optimizer")
    parser.add_argument('-lf', '--loss-fn', type=str, default="mse",
                        choices=LOSS_FN_TYPES.keys(),
                        help="loss function to use to compute loss")
    parser.add_argument('-op', '--optimizer', type=str, default="sgd",
                        choices=OPTIMIZER_TYPES.keys(),
                        help="optimizer to use to learn")


def optimizer_factory(network, training_params: TrainingParameters):
    optimizer = training_params.optimizer

    if optimizer == optim.SGD:
        optimizer = optimizer(network.parameters(), training_params.learning_rate, momentum=0.1)
    else:
        optimizer = optimizer(network.parameters(), training_params.learning_rate)

    return optimizer


def train_network(dataloader: DataLoader, network: nn.Module, training_params: TrainingParameters) -> nn.Module:
    loss_fn = training_params.loss_fn_type()
    optimizer = optimizer_factory(network, training_params)

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


