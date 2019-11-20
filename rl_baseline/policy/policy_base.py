class PolicyBase(object):
    def __init__(self, env, log, tb_writer, args, name):
        super(PolicyBase, self).__init__()

        self.env = env
        self.log = log
        self.tb_writer = tb_writer
        self.args = args
        self.name = name

    def set_dim(self):
        raise NotImplementedError()

    def set_linear_schedule(self):
        raise NotImplementedError()

    def select_stochastic_action(self):
        raise NotImplementedError()

    def clear_memory(self):
        self.memory.clear()

    def set_policy(self):
        if self.args.algorithm == "DQN": 
            from policy.dqn import DQN
            self.policy = DQN(
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                name=self.name,
                args=self.args)

    def save_weight(self, filename, directory):
        self.log[self.args.log_name].info("[{}] Saved weight".format(self.name))
        self.policy.save(filename, directory)

    def load_weight(self, filename, directory="./pytorch_models"):
        self.log[self.args.log_name].info("[{}] Loaded weight".format(self.name))
        self.policy.load(filename, directory)
