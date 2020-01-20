"""Simple replay buffer
Ref: https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
"""
import numpy as np


class ReplayBuffer(object):
    def __init__(self):
        self.storage = []

    def __len__(self):
        return len(self.storage)

    def clear(self):
        self.storage.clear()
        assert len(self.storage) == 0

    def sync(self, memory):
        self.clear()
        for exp in memory.storage:
            self.storage.append(exp)

        assert len(memory) == len(self.storage)

    def add(self, data):
        # Expects tuples of (state, next_state, action, reward, done)
        if len(self.storage) > 1.5e5:
            self.storage.pop(0)
        self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x_gridmap, x_theta = [], []
        y_gridmap, y_theta = [], []
        u, r, d = [], [], []

        for i in ind: 
            X, Y, U, R, D = self.storage[i]

            x_gridmap.append(np.array(X["gridmap"], copy=False))
            x_theta.append(np.array(X["theta"], copy=False))

            y_gridmap.append(np.array(Y["gridmap"], copy=False))
            y_theta.append(np.array(Y["theta"], copy=False))

            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        x_gridmap = np.array(x_gridmap)
        x_theta = np.array(x_theta)

        y_gridmap = np.array(y_gridmap)
        y_theta = np.array(y_theta)

        x = {"gridmap": x_gridmap, "theta": x_theta}
        y = {"gridmap": y_gridmap, "theta": y_theta}

        return \
            x, y, np.asarray(u, dtype=np.int64).reshape(-1, 1), \
            np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
