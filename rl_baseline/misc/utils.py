import torch
import gym
import logging
import numpy as np

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


def set_policy(env, tb_writer, log, args, name):
    from policy.agent import Agent
    policy = Agent(env=env, tb_writer=tb_writer, log=log, name=name, args=args)

    return policy


def preprocess_obs(obs):
    """Preprocess obs
    Grap semantic_gridmap and transpose into torch order: 
    (channel, height, width) assuming (width, height, channel) as input
    """
    obs = obs["semantic_gridmap"]
    return np.swapaxes(obs, 0, 2)
