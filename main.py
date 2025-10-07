import sys
import os
import numpy as np
from envs.grid_env import GridWorld
from agents.random_agent import RandomAgent
from agents.MonteCarloAgent import MonteCarloAgent
from agents.PolicyIteration import PolicyIterationAgent
from agents.QLearningAgent import QLearningAgent
from agents.value_agents import ValueIterationAgent
from trainers.trainer import run_simulation, run_multiple_episodes, run_training_with_plot

def run_agent(env, agent, agent_name, train_needed=True, episodes=1000):
    print(f"\n=== Running simulation for {agent_name} ===")
    
    if train_needed and hasattr(agent, 'train'):
        # Train et récupérer policy + erreurs
        policy, errors = run_training_with_plot(env, agent, episodes=episodes, render_env=False)
        print(f"Policy learned for {agent_name}:\n{policy}")
        # Sauvegarder la table de l’agent
        agent.save_table()
    else:
        # Pour RandomAgent ou agents sans train
        run_simulation(env, agent)
    
    # Lancer plusieurs épisodes pour voir performance
    rewards = run_multiple_episodes(env, agent, episodes=5, render=False)
    print(f"Average reward over 5 episodes: {np.mean(rewards):.2f}")

def main():
    # Paramètres dynamiques
    size = 6
    start_pos = (0, 0)
    goal_positions = [(5,5), (4,2)]          # Exemple : plusieurs objectifs
    obstacles = [(1,1), (2,2), (3,1)]
    max_steps = 50
    cell_size = 60

    # Créer l'environnement dynamique
    env = GridWorld(size=size, start_pos=start_pos, goal_positions=goal_positions,
                    obstacles=obstacles, max_steps=max_steps, cell_size=cell_size)

    # Liste des agents : (instance, nom, train_needed)
    agents = [
        (RandomAgent(env), "RandomAgent", False),
        (MonteCarloAgent(env), "MonteCarloAgent", True),
        (PolicyIterationAgent(env), "PolicyIterationAgent", True),
        (QLearningAgent(env), "QLearningAgent", True),
        (ValueIterationAgent(env), "ValueIterationAgent", True)
    ]

    # Exécuter chaque agent
    for agent, name, train_needed in agents:
        run_agent(env, agent, name, train_needed)

if __name__ == "__main__":
    main()
