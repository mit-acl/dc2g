import numpy as np
from policy.policy_base import PolicyBase
from misc.replay_buffer import ReplayBuffer


class Agent(PolicyBase):
    def __init__(self, env, tb_writer, log, args, name):
        super(Agent, self).__init__(
            env=env, log=log, tb_writer=tb_writer, args=args, name=name)

        self.set_dim()
        self.set_policy()
        self.memory = ReplayBuffer()
        self.epsilon = 1.0  # For exploration

    def set_dim(self):
        self.input_dim = self.env.observation_space.spaces["semantic_gridmap"].shape
        self.output_dim = self.env.action_space.n

        self.log[self.args.log_name].info("[{}] Input dim: {}".format(
            self.name, self.input_dim))
        self.log[self.args.log_name].info("[{}] Output dim: {}".format(
            self.name, self.output_dim))

    def select_deterministic_action(self, obs):
        action = self.policy.select_action(obs)
        assert not np.isnan(action).any()

        return action

    def select_stochastic_action(self, obs, total_timesteps):
        if np.random.rand() > self.epsilon:
            # Exploitation
            action = self.policy.select_action(obs)
        else:
            # Exploration
            action = np.random.randint(low=0, high=self.output_dim, size=(1,))

            if self.epsilon > 0.1:
                self.epsilon *= 0.99999  # Reduce epsilon over time

        assert not np.isnan(action).any()

        self.tb_writer.add_scalar(
            "debug/epsilon", self.epsilon, total_timesteps)

        return action

    def add_memory(self, obs, new_obs, action, reward, done):
        self.memory.add((obs, new_obs, action, reward, done))

    def clear_tmp_memory(self):
        self.tmp_memory.clear()

    def update_policy(self, total_timesteps):
        debug = self.policy.train(
            replay_buffer=self.memory,
            iterations=50)

        self.tb_writer.add_scalars(
            "loss/critic", {self.name: debug["critic_loss"]}, total_timesteps)

    def save(self, episode):
        self.policy.save("critic_" + str(episode), "./pytorch_models")

    def load(self, episode):
        self.policy.load("critic_" + str(episode), "./pytorch_models")
