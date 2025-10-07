import numpy as np

class PolicyIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.random.choice(env.action_space.n, size=(env.size, env.size))
        self.errors = []
        self.trained = False

    def policy_evaluation(self):
        while True:
            delta = 0
            for y in range(self.env.size):
                for x in range(self.env.size):
                    if (x, y) in self.env.obstacles:
                        continue
                    v = self.V[y, x]
                    action = self.policy[y, x]
                    next_state, reward, done, _ = self.env.simulate_step((x, y), action)
                    ny, nx = next_state[1], next_state[0]
                    self.V[y, x] = reward + self.gamma * self.V[ny, nx]
                    delta = max(delta, abs(v - self.V[y, x]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        policy_stable = True
        for y in range(self.env.size):
            for x in range(self.env.size):
                if (x, y) in self.env.obstacles:
                    continue
                old_action = self.policy[y, x]
                action_values = []
                for action in range(self.env.action_space.n):
                    next_state, reward, done, _ = self.env.simulate_step((x, y), action)
                    ny, nx = next_state[1], next_state[0]
                    action_value = reward + self.gamma * self.V[ny, nx]
                    action_values.append(action_value)
                self.policy[y, x] = np.argmax(action_values)
                if old_action != self.policy[y, x]:
                    policy_stable = False
        return policy_stable

    def train(self, episodes=1000):
        if self.trained:
            return self.policy, self.errors
        
        self.errors = []
        print(f"Policy Iteration: Starting training for {episodes} iterations")
        
        for episode in range(episodes):
            self.V_old = self.V.copy()
            self.policy_old = self.policy.copy()
            
            # Policy Evaluation
            self.policy_evaluation()
            
            # Policy Improvement
            policy_stable = self.policy_improvement()
            
            # Calculate error (policy change percentage)
            policy_change = np.mean(self.policy != self.policy_old)
            value_change = np.mean(np.abs(self.V - self.V_old))
            self.errors.append(value_change)
            
            print(f"Policy Iteration Episode {episode}: Policy change = {policy_change:.4f}, Value change = {value_change:.6f}")
            
            if policy_stable:
                print(f"Policy converged after {episode + 1} iterations")
                break
        
        self.trained = True
        return self.policy, self.errors

    def choose_action(self, state):
        y, x = state[1], state[0]
        return self.policy[y, x]

    def save_table(self, policy_file="policy_pi.npy", V_file="V_pi.npy"):
        np.save(policy_file, self.policy)
        np.save(V_file, self.V)

    def load_table(self, policy_file="policy_pi.npy", V_file="V_pi.npy"):
        self.policy = np.load(policy_file)
        self.V = np.load(V_file)
        self.trained = True