import torch
import gym
import logging
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_logger(logger_name, log_file, level=logging.INFO):
    log = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(log_file, mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)

    log.setLevel(level)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler)


def set_log(args):
    log = {}                                                                                                                                        
    set_logger(
        logger_name=args.log_name, 
        log_file=r'{0}{1}'.format("./logs/", args.log_name))
    log[args.log_name] = logging.getLogger(args.log_name)

    return log


def make_env(args):
    import gym_minigrid  # noqa
    env = gym.make(args.env_name)
    env.set_difficulty_level('easy')

    return env


def onehot_from_logits(logits):
    """Given batch of logits, return one-hot sample
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Add a dimension

    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float()
    return argmax_acs


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor):
    """Sample from Gumbel(0, 1)
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if device == torch.device("cuda"):
        tens_type = torch.cuda.FloatTensor
    elif device == torch.device("cpu"):
        tens_type = torch.FloatTensor
    else:
        raise ValueError("Invalid dtype")

    y = logits + sample_gumbel(logits.shape, tens_type=tens_type)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, hard=False):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Ref: https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb
    Ref: https://github.com/shariqiqbal2810/maddpg-pytorch/blob/master/utils/misc.py
    """
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
        if len(logits.shape) == 1:
            logits = logits.unsqueeze(0)  # Add a dimension
    assert len(logits.shape) == 2, "Shape should be: (# of batch, # of action)"

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y = (y_hard - y).detach() + y

    return y
