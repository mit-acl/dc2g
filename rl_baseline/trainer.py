from misc.utils import preprocess_obs

total_timesteps, total_eps = 0, 0


def eval_progress(env, agent, n_eval, log, tb_writer, args):
    global total_eps
    eval_reward = 0.

    for i_eval in range(n_eval):
        ep_timesteps = 0.
        obs = env.reset()
        obs = preprocess_obs(obs)

        while True:
            # Select action
            action = agent.select_deterministic_action(obs)

            # Take action in env
            new_obs, reward, done, _ = env.step(action)
            new_obs = preprocess_obs(new_obs)

            # Add experience to memory
            agent.add_memory(
                obs=obs,
                new_obs=new_obs,
                action=action,
                reward=reward,
                done=False)

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
    obs = env.reset()
    obs = preprocess_obs(obs)

    while True:
        # Select action
        action = agent.select_stochastic_action(obs, total_timesteps)

        # Take action in env
        new_obs, reward, done, _ = env.step(action)
        new_obs = preprocess_obs(new_obs)

        # Add experience to memory
        agent.add_memory(
            obs=obs,
            new_obs=new_obs,
            action=action,
            reward=reward,
            done=False)

        # For next timestep
        obs = new_obs
        ep_timesteps += 1
        total_timesteps += 1
        ep_reward += reward

        if done: 
            total_eps += 1
            log[args.log_name].info("Train episode reward {} at episode {}".format(ep_reward, total_eps))
            tb_writer.add_scalars("reward", {"train_reward": ep_reward}, total_eps)
            tb_writer.add_scalar("debug/memory", len(agent.memory), total_eps)

            return ep_reward


def train(agent, env, log, tb_writer, args):
    while True:
        # Measure performance for reporting results in paper
        if total_eps % 50 == 0:
            eval_progress(env=env, agent=agent, n_eval=1, log=log, tb_writer=tb_writer, args=args)

        # Collect one trajectory
        collect_one_traj(agent=agent, env=env, log=log, args=args, tb_writer=tb_writer)

        # Update policy
        agent.update_policy(total_timesteps=total_timesteps)

        if total_eps % 500 == 0:
            agent.save(total_eps)
