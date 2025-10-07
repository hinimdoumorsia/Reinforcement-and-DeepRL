import numpy as np

class MonteCarloAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.returns_count = np.zeros((env.size, env.size, env.action_space.n))

    def generate_episode(self):
        episode = []
        state = self.env.reset()
        done = False
        while not done:
            action = np.random.choice(self.env.action_space.n)
            next_state, reward, done, _ = self.env.step(action)
            episode.append((state, action, reward))
            state = next_state
        return episode

    def train(self, episodes=1000):
        errors = []
        policy = np.zeros((self.env.size, self.env.size))
        
        for _ in range(episodes):
            episode = self.generate_episode()
            G = 0
            
            for state, action, reward in reversed(episode):
                G = self.gamma * G + reward
                y, x = state[1], state[0]
                self.Q[y, x, action] += G
                self.returns_count[y, x, action] += 1
            
            # Calcul de l’erreur / convergence
            current_policy = np.argmax(self.Q / np.maximum(1, self.returns_count), axis=2)
            error = np.mean(np.abs(current_policy - policy))
            errors.append(error)
            policy = current_policy.copy()
        
        return policy, errors

    def choose_action(self, state):
        y, x = state[1], state[0]
        return np.argmax(self.Q[y, x])

    # ---- Sauvegarde / Chargement ----
    def save_table(self, filename="montecarlo_Q.npy"):
        np.save(filename, self.Q)

    def load_table(self, filename="montecarlo_Q.npy"):
        self.Q = np.load(filename)
