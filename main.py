#!/usr/bin/python3

import argparse
import logging

from torchexperiments.utils.argparseutils import add_training_params_to_parser
from torchexperiments.functions.funcs import (
    get_experiment_from_args,
    FUNCTION_EXPERIMENTS,
)
from torchexperiments.torchutils.dataclasses import print_network_evaluation


def main(args: argparse.Namespace):
    logging.info(f"Started program with the following args: {args}")

    training_params, training_perf, validation_perf = get_experiment_from_args(
        args
    ).run()
    print_network_evaluation(
        args.output_format, training_params, training_perf, validation_perf
    )


def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pytorch-testing")
    add_training_params_to_parser(parser)

    parser.add_argument(
        "function",
        type=str,
        choices=FUNCTION_EXPERIMENTS.keys(),
        help="specify the function to learn",
    )
    parser.add_argument(
        "-ll",
        "--logging-level",
        type=int,
        default=logging.INFO,
        choices=[logging.ERROR, logging.INFO, logging.DEBUG],
        help=f"logging level to use: {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
        + f"{logging.DEBUG}=DEBUG, higher number means less output",
    )
    parser.add_argument(
        "-ts",
        "--training-size",
        type=int,
        default=1000,
        help="specifies how many training data is generated",
    )
    parser.add_argument(
        "-vs",
        "--validation-size",
        type=int,
        default=1000,
        help="specifies how many validation data is generated",
    )
    parser.add_argument(
        "-lo",
        "--logging-output",
        type=str,
        default="stderr",
        help="option to log to file. If option is not specified, all output is sent to stderr",
    )
    parser.add_argument(
        "-of",
        "--output-format",
        type=str,
        default="csv",
        choices=["csv", "human"],
        help="output format, default is csv",
    )
    return parser


def setup_logger(args: argparse.Namespace):
    if args.logging_output == "stderr":
        logging.basicConfig(level=args.logging_level)
    else:
        logging.basicConfig(filename=args.logging_output, level=args.logging_level)


if __name__ == "__main__":
    parser = setup_argument_parser()
    parsed_args = parser.parse_args()
    setup_logger(parsed_args)

    main(parsed_args)
