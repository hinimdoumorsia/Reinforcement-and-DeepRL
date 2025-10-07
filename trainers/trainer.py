import time
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def run_simulation(env, agent, render=True, delay=0.1):
    state = env.reset()
    done = False
    while not done:
        if render:
            env.render()
            time.sleep(delay)
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
    if render:
        env.render()

def run_multiple_episodes(env, agent, episodes=5, render=False, delay=0.1):
    rewards_per_episode = []

    for ep in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            if render:
                env.render()
                time.sleep(delay)
            action = agent.choose_action(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        rewards_per_episode.append(total_reward)
        print(f"Episode {ep+1}: Total reward = {total_reward}")

    return rewards_per_episode

def plot_errors(errors, agent_name="Agent"):
    """Ouvre une fenêtre matplotlib pour tracer la courbe de convergence"""
    plt.figure(figsize=(6,4))
    plt.title(f"Convergence / Erreur - {agent_name}")
    plt.xlabel("Episodes")
    plt.ylabel("Erreur")
    plt.plot(errors, label="Erreur")
    plt.legend()
    plt.grid()
    plt.show()

def run_training_with_plot(env, agent, episodes=1000, render_env=True):
    # Train et récupérer policy + erreurs
    policy, errors = agent.train(episodes=episodes)
    
    # Afficher la courbe d'erreur dans une fenêtre
    plot_errors(errors, agent_name=type(agent).__name__)

    # Optionnel : visualiser une simulation avec la policy apprise
    run_simulation(env, agent, render=render_env)
    
    return policy, errors
