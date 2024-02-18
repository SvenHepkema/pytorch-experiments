from .constants import LOSS_FN_TYPES, OPTIMIZER_TYPES
from .dataclasses import TrainingParameters


def add_training_params_to_parser(parser):
    parser.add_argument(
        "-ep", "--epochs", type=int, default=5000, help="number of epochs to run"
    )
    parser.add_argument(
        "-bs", "--batch-size", type=int, default=100, help="number of samples per batch"
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.001,
        help="learning rate of optimizer",
    )
    parser.add_argument(
        "-ls",
        "--loss-stop",
        type=float,
        default=0.0,
        help="restart the learning process if the first 100 epochs"
        + " result in less loss minimization than the specified percentage. "
        + "If 0 (default), there is no stop and the training will never be restarted",
    )
    parser.add_argument(
        "-lf",
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
        default="sgd",
        choices=OPTIMIZER_TYPES.keys(),
        help="optimizer to use to learn",
    )


def get_training_parameters_from_args(args) -> TrainingParameters:
    return TrainingParameters(
        args.epochs,
        args.learning_rate,
        args.loss_stop,
        LOSS_FN_TYPES[args.loss_fn.lower()],
        OPTIMIZER_TYPES[args.optimizer.lower()],
    )
