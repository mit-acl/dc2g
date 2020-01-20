from misc.utils import preprocess_obs

total_eps = 0


def eval_progress(env, agent, n_eval, log, tb_writer, args):
    global total_eps

    eval_reward = 0.
    ep_timesteps = 0.

    for i_eval in range(n_eval):
        obs = env.reset()
        obs = preprocess_obs(obs)

        while True:
            env.render()

            # Select action
            action = agent.select_stochastic_action(obs, total_timesteps=0)

            # Take action in env
            new_obs, reward, done, _ = env.step(action)
            new_obs = preprocess_obs(new_obs)

            # For next timestep
            obs = new_obs
            eval_reward += reward
            ep_timesteps += 1

            if done:
                break
    eval_reward /= float(n_eval)
    ep_timesteps /= float(n_eval)

    log[args.log_name].info("Evaluation reward {} at episode {}".format(eval_reward, total_eps))
    log[args.log_name].info("Evaluation step {} at episode {}".format(ep_timesteps, total_eps))
    tb_writer.add_scalars("reward", {"eval": eval_reward}, total_eps)
    tb_writer.add_scalars("step", {"eval": ep_timesteps}, total_eps)


def test(agent, env, log, tb_writer, args):
    global total_eps

    # Load weight
    agent.load(episode=4400)
    agent.epsilon = 0.05

    # Perform testing
    while True:
        eval_progress(env=env, agent=agent, n_eval=1, log=log, tb_writer=tb_writer, args=args)
        total_eps += 1
