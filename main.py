#!/usr/bin/python3

import argparse
import logging
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from training import (add_training_params_to_parser,
        DEVICE, TrainingParameters, train_network, show_result)

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

def main(args):
    logging.info(f"Started program with the following args: {args}")
    training_params = TrainingParameters(args)

    data = generate_xor_data(1000)
    labels = [[xor_function(pair)] for pair in data]

    data_tensor = torch.tensor(data, dtype=torch.float32).to(DEVICE)
    labels_tensor = torch.tensor(labels, dtype=torch.float32).to(DEVICE)

    network = Net().to(DEVICE)
    dataloader = DataLoader(list(zip(data_tensor, labels_tensor)), shuffle=True, batch_size=args.batch_size)

    train_network(dataloader, network, training_params)

    validation_data_tensor = torch.FloatTensor(generate_xor_data(1000)).to(DEVICE)
    show_result(validation_data_tensor, network, xor_function)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='pytorch-testing')
    add_training_params_to_parser(parser)

    parser.add_argument('-ll', '--logging-level', type=int, default=logging.INFO,
                        choices=[logging.ERROR, logging.INFO, logging.DEBUG],
                        help=f"logging level to use: {logging.ERROR}=ERROR, {logging.INFO}=INFO, "
                        +f"{logging.DEBUG}=DEBUG, higher number means less output")
    parser.add_argument('-lo', '--logging-output', type=str, default="stderr",
                        help="option to log to file. If option is not specified, all output is sent to stderr")


    parsed_args = parser.parse_args()

    if parsed_args.logging_output == "stderr":
        logging.basicConfig(level=parsed_args.logging_level)
    else:
        logging.basicConfig(filename=parsed_args.logging_output, level=parsed_args.logging_level)

    main(parsed_args)
