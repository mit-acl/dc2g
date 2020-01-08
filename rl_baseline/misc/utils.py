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
    Additionally, obtain pos and theta
    """
    # Get gridmap
    # gridmap = np.swapaxes(obs["semantic_gridmap"], 0, 2)
    gridmap = np.zeros((3, 50, 50))

    # Normalize pos
    pos_x, pos_y = obs["pos"]
    pos_x = (pos_x - 25) / 50.
    pos_y = (pos_y - 25) / 50.
    pos = np.array([pos_x, pos_y])
    assert pos_x <= 1.
    assert pos_x >= -1.
    assert pos_y <= 1.
    assert pos_y >= -1.

    # Normalize theta
    theta = np.array(obs["theta"] / 3.5)
    assert theta <= 1.
    assert theta >= -1.

    return {"gridmap": gridmap, "pos": pos, "theta": theta}
