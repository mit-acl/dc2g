import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize, SubprocVecEnv
from stable_baselines import PPO2
from stable_baselines import A2C
from stable_baselines import ACKTR
from stable_baselines.common import set_global_seeds
import datetime


import gym_minigrid

env = gym.make('MiniGrid-EmptySLAM-32x32-v0')
env.set_difficulty_level('easy')
env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
env = VecNormalize(env, norm_obs=True, norm_reward=False,
                   clip_obs=10.)

# def make_env(env_id, rank, seed=0):
#     """
#     Utility function for multiprocessed env.

#     :param env_id: (str) the environment ID
#     :param num_env: (int) the number of environments you wish to have in subprocesses
#     :param seed: (int) the inital seed for RNG
#     :param rank: (int) index of the subprocess
#     """
#     def _init():
#         env = gym.make(env_id)
#         env.seed(seed + rank)
#         # if env_id == 'MiniGrid-EmptySLAM-32x32-v0':
#         env.set_difficulty_level('easy')
#         return env
#     set_global_seeds(seed)
#     return _init

# # env_id = "CartPole-v1"
# env_id = "MiniGrid-EmptySLAM-32x32-v0"
# num_cpu = 4  # Number of processes to use
# # Create the vectorized environment
# env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

# # Custom MLP policy of two layers of size 32 each with tanh activation function
policy_kwargs = dict(net_arch=[32, 32])
# # model = ACKTR(MlpPolicy, env, verbose=1)
model = A2C("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs, tensorboard_log='/home/mfe/tb_log', learning_rate=2e-3)
# # model.learn(total_timesteps=100000)
# model.learn(total_timesteps=10000)
# # model.learn(total_timesteps=100)
# model.save("a2c_lunar_{:%Y-%m-%d_%H_%M_%S}".format(datetime.datetime.now()))

model.load("a2c_lunar_2019-02-26_21_39_22.pkl")

# env = gym.make(env_id)
# env = DummyVecEnv([lambda: env])  # The algorithms require a vectorized environment to run
obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()