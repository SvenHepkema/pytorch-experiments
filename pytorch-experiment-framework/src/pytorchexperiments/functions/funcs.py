import argparse

from pytorchexperiments.functions.experiment import Experiment
from pytorchexperiments.functions.andf import get_and_experiment
from pytorchexperiments.functions.xor import get_xor_experiment

FUNCTION_EXPERIMENTS = {
    "and": get_and_experiment,
    "xor": get_xor_experiment,
}


def get_experiment_from_args(args: argparse.Namespace) -> Experiment:
    return FUNCTION_EXPERIMENTS[args.function.lower()](args)
