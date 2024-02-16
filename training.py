from collections.abc import Callable
from dataclasses import dataclass
import logging
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

LOSS_FN_TYPES = {
    "l1": nn.L1Loss,
    "smoothl1": nn.SmoothL1Loss,
    "mse": nn.MSELoss,
    # "ctc": nn.CTCLoss, FIX: requires 2 positional arguments
    "nll": nn.NLLLoss,
    "poissonnll": nn.PoissonNLLLoss,
    # "gaussiannll": nn.GaussianNLLLoss, FIX requires a positional argument
    "kldiv": nn.KLDivLoss,
    "bce": nn.BCELoss,
    "bcelogits": nn.BCEWithLogitsLoss,
    # "marginranking": nn.MarginRankingLoss, FIX requires a positional argument
    "hingeembedding": nn.HingeEmbeddingLoss,
    "multimargin": nn.MultiMarginLoss,
    "multilabelmargin": nn.MultiLabelMarginLoss,
    "multilabelsoftmargin": nn.MultiLabelSoftMarginLoss,
    "huber": nn.HuberLoss,
    "softmargin": nn.SoftMarginLoss,
    # "cosineembedding": nn.CosineEmbeddingLoss, FIX requires a positional argument
    "crossentropy": nn.CrossEntropyLoss,
    # "tripletmargin": nn.TripletMarginLoss, FIX requires a positional argument
    # "tripletmargindistance": nn.TripletMarginWithDistanceLoss, FIX requires a positional argument
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
    # "lbfgs": optim.LBFGS,  FIX: requires a positional argument
    "nadam": optim.NAdam,
    "radam": optim.RAdam,
    "rmsprop": optim.RMSprop,
    "rprop": optim.Rprop,
}

@dataclass(frozen=True)
class TrainingPerformance:
    training_time: float
    first_loss: float
    last_loss: float

    def get_as_csv_string(self) -> str:
        return (f"{self.first_loss},{self.last_loss}," 
                +f"{self.first_loss-self.last_loss:.15f},{self.training_time},")


@dataclass(frozen=True)
class ValidationPerformance:
    validation_count: int
    correct_validation_count: int

    def get_as_csv_string(self) -> str:
        return f"{self.validation_count},{self.correct_validation_count},"


@dataclass
class TrainingParameters:
    epochs: int
    learning_rate: float
    loss_stop: float
    loss_fn_type: Callable
    optimizer: Callable

    def __init__(self, args):
        self.epochs = args.epochs
        self.learning_rate = args.learning_rate
        self.loss_stop = args.loss_stop
        self.loss_fn_type = LOSS_FN_TYPES[args.loss_fn.lower()]
        self.optimizer = OPTIMIZER_TYPES[args.optimizer.lower()]

    def get_as_csv_string(self) -> str:
        def get_key_with_value(dictionary, value):
            return list(dictionary.keys())[list(dictionary.values()).index(value)]

        return (f"{self.epochs},{self.learning_rate},"
            +f"{get_key_with_value(LOSS_FN_TYPES, self.loss_fn_type)},"
            +f"{get_key_with_value(OPTIMIZER_TYPES, self.optimizer)},")


def add_training_params_to_parser(parser):
    parser.add_argument('-ep', '--epochs', type=int, default=5000,
                        help="number of epochs to run")
    parser.add_argument('-bs', '--batch-size', type=int, default=100,
                        help="number of samples per batch")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001,
                        help="learning rate of optimizer")
    parser.add_argument('-ls', '--loss-stop', type=float, default=0.0,
                        help="restart the learning process if the first 100 epochs"
                        +" result in less loss minimization than the specified percentage. "
                        +"If 0 (default), there is no stop and the training will never be restarted")
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


def train_network(dataloader: DataLoader, network: nn.Module, 
                  training_params: TrainingParameters) -> TrainingPerformance | None:
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
                if first_loss * (1-training_params.loss_stop) < running_loss:
                    return None

    end_time = time.time() 

    return TrainingPerformance(end_time - start_time, first_loss, running_loss)


def evaluate_network(validation_data: torch.Tensor, network: nn.Module, 
                     evaluator: Callable) -> ValidationPerformance:
    """ Returns the number of correctly predicted labels for each record in validation data."""

    correct_count = 0
    output = network(validation_data)
    results = list(zip(validation_data, output))
    for validation_data, output in results:
        correct = int(round(evaluator(validation_data))) == int(round(output.item()))
        correct_count += int(correct)
        logging.debug(f"{validation_data} \t=>\t {output.item()} | {correct}")

    return ValidationPerformance(len(results), correct_count)


def print_network_evaluation_as_human_readable(
        training_perf: TrainingPerformance, 
        validation_perf: ValidationPerformance):
    percentage_correct = (validation_perf.correct_validation_count
                          /validation_perf.validation_count*100)

    print("")
    print("")
    print("====================")
    print("Finished training...")
    print(f"{validation_perf.correct_validation_count}/{validation_perf.validation_count} correct "
          +f"({percentage_correct:.1f}%) in {training_perf.training_time}s")


def print_network_evaluation_as_csv(training_params: TrainingParameters, 
                                    training_perf: TrainingPerformance,
                                    validation_perf: ValidationPerformance):
    print(training_params.get_as_csv_string() 
            + training_perf.get_as_csv_string()
            + validation_perf.get_as_csv_string())
