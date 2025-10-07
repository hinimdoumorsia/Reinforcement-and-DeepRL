import numpy as np

class MonteCarloAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.returns_sum = np.zeros((env.size, env.size, env.action_space.n))
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
            if len(episode) > 100:  # Éviter les épisodes infinis
                break
        return episode

    def train(self, episodes=1000):
        errors = []
        
        for episode_num in range(episodes):
            # Générer un épisode
            episode = self.generate_episode()
            G = 0
            
            # Parcourir l'épisode à l'envers
            for t in range(len(episode)-1, -1, -1):
                state, action, reward = episode[t]
                G = self.gamma * G + reward
                y, x = state[1], state[0]
                
                # First-visit MC
                if (state, action) not in [(s,a) for s,a,r in episode[:t]]:
                    self.returns_sum[y, x, action] += G
                    self.returns_count[y, x, action] += 1
                    self.Q[y, x, action] = self.returns_sum[y, x, action] / self.returns_count[y, x, action]
            
            # Calculer l'erreur (variation de la politique)
            if episode_num > 0:
                policy_change = np.mean(np.abs(np.argmax(self.Q, axis=2) - np.argmax(self.Q_old, axis=2)))
                errors.append(policy_change)
            else:
                errors.append(1.0)
                
            self.Q_old = self.Q.copy()
        
        policy = np.argmax(self.Q, axis=2)
        return policy, errors

    def choose_action(self, state):
        y, x = state[1], state[0]
        if np.all(self.Q[y, x] == 0):
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.Q[y, x])

    def save_table(self, filename="montecarlo_Q.npy"):
        np.save(filename, self.Q)

    def load_table(self, filename="montecarlo_Q.npy"):
        self.Q = np.load(filename)