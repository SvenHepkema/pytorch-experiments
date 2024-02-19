import argparse

from torchexperiments.torchutils.constants import LOSS_FN_TYPES, OPTIMIZER_TYPES
from torchexperiments.torchutils.dataclasses import TrainingParameters


def add_training_params_to_parser(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-ep", "--epochs", type=int, default=5000, help="number of epochs to run"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=100, help="number of samples per batch"
    )
    parser.add_argument(
        "-lr",
        "--learning-rate",
        type=float,
        default=0.001,
        help="learning rate of optimizer",
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
        help="loss function to use to compute loss",
    )
    parser.add_argument(
        "-op",
        "--optimizer",
        type=str,
        default="adam",
        choices=OPTIMIZER_TYPES.keys(),
        help="optimizer to use to learn",
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
