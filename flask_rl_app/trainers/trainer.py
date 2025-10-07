import numpy as np

def run_simulation_step(env, agent):
    state = env.reset()
    done = False
    positions = []
    errors = []

    # Si l'agent a un train() qui renvoie erreur (MonteCarlo, QLearning, Policy/Value Iteration)
    if hasattr(agent, "train"):
        policy, train_errors = agent.train(episodes=1)  # on peut ajuster le nombre d'itérations
    else:
        train_errors = []

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # Calcul de l'erreur si disponible
        if hasattr(agent, "Q"):
            current_policy = np.argmax(agent.Q, axis=2)
            error = np.mean(np.abs(current_policy - current_policy))  # simplifié, placeholder
        elif hasattr(agent, "V"):
            current_policy = getattr(agent, "policy", None)
            error = 0 if current_policy is None else np.mean(np.abs(current_policy - current_policy))
        else:
            error = 0

        positions.append(list(next_state))
        errors.append(error)
        state = next_state

        yield {
            "state": list(next_state),
            "reward": reward,
            "done": done,
            "error": error
        }

def run_multiple_episodes(env, agent, episodes=5):
    all_positions = []
    all_errors = []

    for ep in range(episodes):
        episode_positions = []
        episode_errors = []

        for step_info in run_simulation_step(env, agent):
            episode_positions.append(step_info["state"])
            episode_errors.append(step_info["error"])

        all_positions.append(episode_positions)
        all_errors.append(episode_errors)

    return all_positions, all_errors
