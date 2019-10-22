"""Modified Twin Delayed Deep Deterministic Policy Gradients (TD3)
TD3 Ref: https://github.com/sfujim/TD3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, args, name):
        super(Critic, self).__init__()
        width, height, channel = self._decompose_input_dim(input_dim)

        setattr(self, name + "_conv1", nn.Conv2d(channel, 16, kernel_size=5, stride=2))
        setattr(self, name + "_bn1", nn.BatchNorm2d(16))
        setattr(self, name + "_conv2", nn.Conv2d(16, 32, kernel_size=5, stride=2))
        setattr(self, name + "_bn2", nn.BatchNorm2d(32))
        setattr(self, name + "_conv3", nn.Conv2d(32, 32, kernel_size=5, stride=2))
        setattr(self, name + "_bn3", nn.BatchNorm2d(32))

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # Ref: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        for i_conv in range(3):
            kernel_size = getattr(self, name + "_conv" + str(i_conv + 1)).kernel_size[0]
            stride = getattr(self, name + "_conv" + str(i_conv + 1)).stride[0]
            width = self._conv2d_size_out(width, kernel_size, stride)

        for i_conv in range(3):
            kernel_size = getattr(self, name + "_conv" + str(i_conv + 1)).kernel_size[0]
            stride = getattr(self, name + "_conv" + str(i_conv + 1)).stride[0]
            height = self._conv2d_size_out(height, kernel_size, stride)

        linear_input_size = width * height * 32
        setattr(self, name + "_fc", nn.Linear(linear_input_size, output_dim))

        self.name = name

    def forward(self, x):
        batch_size = x.shape[0]
        for i_conv in range(3):
            x = getattr(self, self.name + "_conv" + str(i_conv + 1))(x)
            x = getattr(self, self.name + "_bn" + str(i_conv + 1))(x)
            x = F.relu(x)

        x = x.view(batch_size, -1)  # Concat output of conv
        return getattr(self, self.name + "_fc")(x)

    def _decompose_input_dim(self, input_dim):
        """Decompose input dim w.r.t. (width, height, channel)
        Assuming (width, height, channel) as input"""
        assert len(input_dim) == 3
        width = input_dim[0]
        height = input_dim[1]
        channel = input_dim[2]

        return width, height, channel

    def _conv2d_size_out(self, size, kernel_size, stride):
        """Computes output dimension from convolutional layer"""
        return (size - (kernel_size - 1) - 1) // stride + 1


class DQN(object):
    def __init__(self, input_dim, output_dim, name, args):
        self.critic = Critic(input_dim, output_dim, args, name=name + "_critic").to(device)
        self.critic_target = Critic(input_dim, output_dim, args, name=name + "_critic").to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.name = name
        self.args = args

    def select_action(self, state, display=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)  # Add a batch dimension

        logits = self.critic(state)
        logits = logits.cpu().data.numpy()
        logits = np.squeeze(logits, axis=0)
        if display:
            print(logits)

        return np.argmax(logits)

    def train(self, replay_buffer, iterations):
        debug = {}
        debug["critic_loss"] = 0.

        for it in range(iterations):
            # Sample replay buffer 
            x, y, u, r, d = replay_buffer.sample(self.args.batch_size)
            state = torch.FloatTensor(x).to(device)
            next_state = torch.FloatTensor(y).to(device)
            action = torch.LongTensor(u).to(device)
            reward = torch.FloatTensor(r).to(device)
            done = torch.FloatTensor(1 - d).to(device)

            # Get target Q value
            next_state_Q = self.critic_target(next_state).max(1)[0].unsqueeze(1)
            target_Q = reward + (done * self.args.discount * next_state_Q).detach()
            
            # Get current Q value
            state_Q = self.critic(state).gather(1, action)

            # Compute critic loss
            critic_loss = F.smooth_l1_loss(state_Q, target_Q)

            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
            self.critic_optimizer.step()
            debug["critic_loss"] += critic_loss.cpu().data.numpy().flatten()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.args.tau * param.data + (1 - self.args.tau) * target_param.data)

        return debug

    def save(self, filename, directory):
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))

    def load(self, filename, directory):
        from collections import OrderedDict

        critic_weight = torch.load('%s/%s_critic.pth' % (directory, filename), map_location='cpu')

        critic_weight_fixed = OrderedDict()
        for k, v in critic_weight.items():
            name_fixed = self.name
            for i_name, name in enumerate(k.split("_")):
                if i_name > 0:
                    name_fixed += "_" + name
            critic_weight_fixed[name_fixed] = v

        self.critic.load_state_dict(critic_weight_fixed)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
