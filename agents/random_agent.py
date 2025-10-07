import random
import numpy as np

class RandomAgent:
    def __init__(self, env=None):
        # Si on passe l'environnement, on peut récupérer size/action_space
        if env is not None:
            self.action_space = env.action_space.n
            self.size = env.size
        else:
            self.action_space = 4
            self.size = None
        # Créer une "table" vide pour compatibilité avec sauvegarde
        self.table = None

    def choose_action(self, state=None):
        return random.randint(0, self.action_space - 1)

    def train(self, episodes=1000):
        """Méthode train pour compatibilité avec les autres agents"""
        # Pour RandomAgent, on retourne des erreurs nulles
        errors = [0] * episodes
        policy = np.zeros((self.size, self.size)) if self.size else None
        return policy, errors

    # ---- Sauvegarde / Chargement (compatible même si vide) ----
    def save_table(self, filename="random_agent_table.npy"):
        # Créer table vide si nécessaire
        if self.table is None and self.size is not None:
            self.table = np.zeros((self.size, self.size, self.action_space))
        np.save(filename, self.table)

    def load_table(self, filename="random_agent_table.npy"):
        self.table = np.load(filename)