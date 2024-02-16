#!/usr/bin/python3

from collections.abc import Callable
import argparse
import logging
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training import (add_training_params_to_parser, print_network_evaluation,
        DEVICE, evaluate_network, get_training_parameters_from_args, train_network)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2, 60) 
        self.fc2 = nn.Linear(60, 1)
        self.rl1 = nn.ReLU()
        self.rl2 = nn.ReLU()
	
    def forward(self, x):
        x = self.fc1(x)
        x = self.rl1(x)
        x = self.fc2(x)
        x = self.rl2(x)

        return x


def xor_function(gates: tuple):
    a = gates[0]
    b = gates[1]

    if (a < 0.5 and b > 0.5) or (a > 0.5 and b < 0.5):
        return 1.0
    else:
        return 0.0


def generate_xor_data(n: int):
    data = []

    for _ in range(n):
        data.append((random.uniform(0, 1), random.uniform(0, 1)))

    return data


def generate_dataloader(data_generator: Callable, data_evaluator: Callable,
                        training_size: int, batch_size: int) -> DataLoader:

    data = data_generator(training_size)
    labels = [[data_evaluator(pair)] for pair in data]

    data_tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

    return DataLoader(list(zip(data_tensor, labels_tensor)), shuffle=True, batch_size=batch_size)


def generate_network() -> nn.Module:
    return Net().to(DEVICE)


def main(args):
    logging.info(f"Started program with the following args: {args}")

    training_params = get_training_parameters_from_args(args)
    training_dataloader = generate_dataloader(generate_xor_data, xor_function, args.training_size, args.batch_size)
    training_perf, network = train_network(training_dataloader, training_params, generate_network)

    validation_data_tensor = torch.FloatTensor(generate_xor_data(args.validation_size)).to(DEVICE)
    validation_perf = evaluate_network(validation_data_tensor, network, xor_function)

    print_network_evaluation(args.output_format, training_params, training_perf, validation_perf)



def setup_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog='pytorch-testing')
    add_training_params_to_parser(parser)

    parser.add_argument('-ll', '--logging-level', type=int, default=logging.INFO,
                        choices=[logging.ERROR, logging.INFO, logging.DEBUG],
                        help=f"logging level to use: {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
                        +f"{logging.DEBUG}=DEBUG, higher number means less output")
    parser.add_argument('-ts', '--training-size', type=int, default=1000,
                        help="specifies how many training data is generated")
    parser.add_argument('-vs', '--validation-size', type=int, default=1000,
                        help="specifies how many validation data is generated")
    parser.add_argument('-lo', '--logging-output', type=str, default="stderr",
                        help="option to log to file. If option is not specified, all output is sent to stderr")
    parser.add_argument('-of', '--output-format', type=str, default="csv", choices=["csv","human"],
                        help="output format, default is csv")
    return parser


def setup_logger(arguments):
    if arguments.logging_output == "stderr":
        logging.basicConfig(level=arguments.logging_level)
    else:
        logging.basicConfig(filename=arguments.logging_output, level=arguments.logging_level)


if __name__ == "__main__":
    parser = setup_argument_parser()
    parsed_args = parser.parse_args()
    setup_logger(parsed_args)

    main(parsed_args)
