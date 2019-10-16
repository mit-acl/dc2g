import copy
import numpy as np 

total_timesteps = 0
total_eps = 0


def eval_progress(env, agent, n_eval, log, tb_writer, args):
    global total_eps
    eval_reward = 0.

    for i_eval in range(n_eval):
        print("i_eval:", i_eval)
        ep_timesteps = 0.
        obs = env.reset()

        while True:
            # Select action
            action = agent.select_deterministic_action(obs)
            print("action:", action)

            # Take action in env
            new_obs, reward, done, _ = env.step(action)

            # For next timestep
            obs = new_obs
            eval_reward += reward
            ep_timesteps += 1

            if done:
                break
    eval_reward /= float(n_eval)

    log[args.log_name].info("Evaluation Reward {} at episode {}".format(eval_reward, total_eps))
    tb_writer.add_scalars("reward", {"eval_reward": eval_reward}, total_eps)


def collect_one_traj(agent, env, log, args, tb_writer):
    global total_timesteps, total_eps

    ep_reward = 0
    ep_timesteps = 0
    env_observation = env.reset()

    while True:
        # Select action
        agent_action = agent.select_stochastic_action(np.array(env_observation), total_timesteps)

        # Take action in env
        new_env_observation, env_reward, done, _ = env.step(copy.deepcopy(agent_action))

        # Add experience to memory
        agent.add_memory(
            obs=env_observation,
            new_obs=new_env_observation,
            action=agent_action,
            reward=env_reward,
            done=done)

        # For next timestep
        env_observation = new_env_observation
        ep_timesteps += 1
        total_timesteps += 1
        ep_reward += env_reward

        if done: 
            total_eps += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_eps))
            tb_writer.add_scalars("reward", {"train_reward": ep_reward}, total_eps)

            return ep_reward


def train(agent, env, log, tb_writer, args):
    while True:
        # Measure performance for reporting results in paper
        eval_progress(env=env, agent=agent, n_eval=10, log=log, tb_writer=tb_writer, args=args)
        import sys
        sys.exit()

        # Collect one trajectory
        collect_one_traj(agent=agent, env=env, log=log, args=args, tb_writer=tb_writer)
        tb_writer.add_scalar("debug/memory", len(agent.memory), total_eps)

        # Update policy
        agent.update_policy(total_timesteps=total_timesteps)
