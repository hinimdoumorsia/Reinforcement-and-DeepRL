import random
import numpy as np

class RandomAgent:
    def __init__(self, env=None):
        self.env = env
        if env is not None:
            self.action_space = env.action_space.n
            self.size = env.size
        else:
            self.action_space = 4
            self.size = None
        self.Q = np.zeros((env.size, env.size, env.action_space.n)) if env else None

    def choose_action(self, state=None):
        return random.randint(0, self.action_space - 1)

    def train(self, episodes=1000):
        """Méthode train pour compatibilité"""
        errors = []
        for episode in range(episodes):
            # Pour Random, on simule un entraînement basique
            error = 0.1 * (1 - episode/episodes)  # Erreur qui diminue légèrement
            errors.append(error)
        return None, errors

    def save_table(self, filename="random_agent.npy"):
        if self.Q is not None:
            np.save(filename, self.Q)

    def load_table(self, filename="random_agent.npy"):
        self.Q = np.load(filename)