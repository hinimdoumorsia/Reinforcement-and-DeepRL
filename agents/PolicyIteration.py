import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9):
        self.env = env
        self.gamma = gamma
        self.policy = np.random.choice(env.action_space.n, size=(env.size, env.size))
        self.V = np.zeros((env.size, env.size))

    def train(self, episodes=1000):
        errors = []
        for _ in range(episodes):
            # Policy Evaluation
            delta = 0
            for y in range(self.env.size):
                for x in range(self.env.size):
                    v = self.V[y, x]
                    action = self.policy[y, x]
                    next_state, reward, done, _ = self.env.simulate_step((x, y), action)
                    ny, nx = next_state[1], next_state[0]
                    self.V[y, x] = reward + self.gamma * self.V[ny, nx]
                    delta = max(delta, abs(v - self.V[y, x]))
            # Policy Improvement
            policy_stable = True
            for y in range(self.env.size):
                for x in range(self.env.size):
                    old_action = self.policy[y, x]
                    q_values = []
                    for a in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step((x, y), a)
                        ny, nx = next_state[1], next_state[0]
                        q_values.append(reward + self.gamma * self.V[ny, nx])
                    self.policy[y, x] = np.argmax(q_values)
                    if old_action != self.policy[y, x]:
                        policy_stable = False
            errors.append(delta)
        return self.policy, errors

    # ---- Sauvegarde / Chargement ----
    def save_table(self, policy_file="policy_PI.npy", V_file="V_PI.npy"):
        np.save(policy_file, self.policy)
        np.save(V_file, self.V)

    def load_table(self, policy_file="policy_PI.npy", V_file="V_PI.npy"):
        self.policy = np.load(policy_file)
        self.V = np.load(V_file)
