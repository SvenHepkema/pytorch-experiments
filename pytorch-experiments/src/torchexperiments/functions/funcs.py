import argparse

from torchexperiments.functions.experiment import Experiment
from torchexperiments.functions.andf import get_and_experiment
from torchexperiments.functions.max import get_max_experiment
from torchexperiments.functions.xor import get_xor_experiment
from torchexperiments.functions.avg import get_avg_experiment

FUNCTION_EXPERIMENTS = {
    "and": get_and_experiment,
    "xor": get_xor_experiment,
    "avg": get_avg_experiment,
    "max": get_max_experiment,
}


def get_experiment_from_args(args: argparse.Namespace) -> Experiment:
    return FUNCTION_EXPERIMENTS[args.function.lower()](args)
