import torch
import torch.nn as nn
import torch.optim as optim

ARGPARSE_USE_DEFAULT_STR = "use_default"

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# From https://pytorch.org/docs/stable/nn.html#loss-functions
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

# From https://pytorch.org/docs/stable/optim.html#algorithms
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

# From https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity
ACTIVATION_TYPES = {
    "elu": torch.nn.ELU,
    "relu": torch.nn.ReLU,
    "silu": torch.nn.SiLU,
    "sigmoid": torch.nn.Sigmoid,
    "softplus": torch.nn.Softplus,
    "softmax": torch.nn.Softmax,
    "softmin": torch.nn.Softmin,
}
