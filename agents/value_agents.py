import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)

    def train(self, episodes=1000):
        errors = []
        for _ in range(episodes):
            delta = 0
            for y in range(self.env.size):
                for x in range(self.env.size):
                    v = self.V[y, x]
                    q_values = []
                    for a in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step((x, y), a)
                        ny, nx = next_state[1], next_state[0]
                        q_values.append(reward + self.gamma * self.V[ny, nx])
                    self.V[y, x] = max(q_values)
                    delta = max(delta, abs(v - self.V[y, x]))
            # Update policy
            for y in range(self.env.size):
                for x in range(self.env.size):
                    q_values = []
                    for a in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step((x, y), a)
                        ny, nx = next_state[1], next_state[0]
                        q_values.append(reward + self.gamma * self.V[ny, nx])
                    self.policy[y, x] = np.argmax(q_values)
            errors.append(delta)
        return self.policy, errors

    # ---- Sauvegarde / Chargement ----
    def save_table(self, policy_file="policy_VI.npy", V_file="V_VI.npy"):
        np.save(policy_file, self.policy)
        np.save(V_file, self.V)

    def load_table(self, policy_file="policy_VI.npy", V_file="V_VI.npy"):
        self.policy = np.load(policy_file)
        self.V = np.load(V_file)
