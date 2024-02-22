import argparse

from torchexperiments.torchutils.constants import (
    ACTIVATION_TYPES,
    ARGPARSE_USE_DEFAULT_STR,
    LOSS_FN_TYPES,
    OPTIMIZER_TYPES,
)
from torchexperiments.torchutils.dataclasses import (
    NetworkParameters,
    TrainingParameters,
)


def add_training_params_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ep",
        "--epochs",
        type=int,
        default=5000,
        help="number of epochs to run (default=5000)",
    )
    parser.add_argument(
        "-bs",
        "--batch-size",
        type=int,
        default=100,
        help="number of samples per batch (default=100)",
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate of optimizer (default=0.001)",
    )
    parser.add_argument(
        "-lor",
        "--loss-restart",
        type=float,
        default=0.0,
        help="restart the learning process if the epochs in the first interval"
        + " result in less loss minimization than the specified percentage. "
        + "If 0 (default), there is no restart and the training will never be restarted",
    )
    parser.add_argument(
        "-los",
        "--loss-stop",
        type=float,
        default=0.0,
        help="stop the learning process if there is less loss minimization"
        + " than the specified percentage. "
        + "If 0 (default), there is no stop and the training will never be stopped",
    )
    parser.add_argument(
        "-lof",
        "--loss-fn",
        type=str,
        default="mse",
        choices=LOSS_FN_TYPES.keys(),
        help="loss function to use to compute loss (default=mse)",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="adam",
        choices=OPTIMIZER_TYPES.keys(),
        help="optimizer to use to learn (default=adam)",
    )
    parser.add_argument(
        "-ei",
        "--epoch-interval",
        type=int,
        default=100,
        help="interval in which the loss conditions is checked and the"
        + " current running loss is logged during the training process",
    )


def get_training_parameters_from_args(args: argparse.Namespace) -> TrainingParameters:
    return TrainingParameters(
        args.epochs,
        args.learning_rate,
        args.loss_restart,
        args.loss_stop,
        args.epoch_interval,
        LOSS_FN_TYPES[args.loss_fn.lower()],
        OPTIMIZER_TYPES[args.optimizer.lower()],
    )


def add_network_params_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-s1l",
        "--size-first-layer",
        type=int,
        default=-1,
        help="Size of first middle layer. If -1 (default), the network uses it's own default.",
    )
    parser.add_argument(
        "-a1l",
        "--activation-first-layer",
        type=str,
        default=ARGPARSE_USE_DEFAULT_STR,
        choices=list(ACTIVATION_TYPES.keys()) + [ARGPARSE_USE_DEFAULT_STR],
        help="Activation function of first middle layer. If not specified, the math function uses it's own default.",
    )
    parser.add_argument(
        "-s2l",
        "--size-second-layer",
        type=int,
        default=-1,
        help="Size of second middle layer. If -1 (default), the network uses it's own default. If 0, no second middle layer is used",
    )
    parser.add_argument(
        "-a2l",
        "--activation-second-layer",
        type=str,
        default=ARGPARSE_USE_DEFAULT_STR,
        choices=list(ACTIVATION_TYPES.keys()) + [ARGPARSE_USE_DEFAULT_STR],
        help="Activation function of second middle layer. If not specified, the math function uses it's own default.",
    )


def get_network_parameters_from_args(args: argparse.Namespace) -> NetworkParameters:
    return NetworkParameters(
        args.size_first_layer,
        args.activation_first_layer,
        args.size_second_layer,
        args.activation_second_layer,
    )
