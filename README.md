# pytorch-tests

In this repository I test pytorch neural networks by testing the accuracy and learning speed of neural networks when they need to learn a mathematical function. For analysis purposes, there are scripts that use [GNU Parallel](https://www.gnu.org/software/parallel/) to generate the data and scripts that use [gnuplot](http://www.gnuplot.info/) to visualize the data.

# Usage

Run the `main.py` file with any of the following parameters:

- `-h`, `--help` Show all available parameters.
- `-ep`, `--epochs` Number of epochs to run.
- `-bs`, `--batch-size` Number of samples per batch.
- `-lr`, `--learning_rate` Learning rate of optimizer.
- `-lf`, `--loss-fn` Loss function to use to compute loss.
- `-op`, `--optimizer` Optimizer to use to learn.
- `-ll`, `--logging-level` Logging level to use: 40=ERROR, 20=INFO, 10=DEBUG, higher number means less output.
- `-lo`, `--logging-output` Option to log to file. If option is not specified, all output is sent to stderr.

# Loss functions

The following loss functions are currently supported, for a description of each function see [this pytorch page](https://pytorch.org/docs/stable/nn.html#loss-functions).

- l1
- smoothl1
- mse
- nll
- poissonnll
- kldiv
- bce
- bcelogits
- hingeembedding
- multimargin
- multilabelmargin
- multilabelsoftmargin
- huber
- softmargin
- crossentropy


# Optimizers

The following optimizers are currently supported, for a description of each algorithm see [this pytorch page](https://pytorch.org/docs/stable/optim.html#algorithms).

- adadelta
- adagrad
- adam
- adamw
- adamsparse
- adamax
- asgd
- sgd
- nadam
- radam
- rmsprop
- rprop
