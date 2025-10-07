import numpy as np

class ValueIterationAgent:
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        self.V = np.zeros((env.size, env.size))
        self.policy = np.zeros((env.size, env.size), dtype=int)
        self.errors = []
        self.trained = False

    def train(self, episodes=1000):
        if self.trained:
            return self.policy, self.errors
        
        self.errors = []
        print(f"Value Iteration: Starting training for {episodes} iterations")
        
        for episode in range(episodes):
            delta = 0
            V_old = self.V.copy()
            
            for y in range(self.env.size):
                for x in range(self.env.size):
                    if (x, y) in self.env.obstacles:
                        continue
                    
                    v_old = self.V[y, x]
                    action_values = []
                    
                    for action in range(self.env.action_space.n):
                        next_state, reward, done, _ = self.env.simulate_step((x, y), action)
                        ny, nx = next_state[1], next_state[0]
                        action_value = reward + self.gamma * self.V[ny, nx]
                        action_values.append(action_value)
                    
                    self.V[y, x] = max(action_values)
                    delta = max(delta, abs(v_old - self.V[y, x]))
            
            self.errors.append(delta)
            print(f"Value Iteration Episode {episode}: Delta = {delta:.6f}")
            
            if delta < self.theta:
                print(f"Value Iteration converged after {episode + 1} iterations")
                break

        # Extract optimal policy after value iteration
        print("Extracting optimal policy...")
        for y in range(self.env.size):
            for x in range(self.env.size):
                if (x, y) in self.env.obstacles:
                    continue
                
                action_values = []
                for action in range(self.env.action_space.n):
                    next_state, reward, done, _ = self.env.simulate_step((x, y), action)
                    ny, nx = next_state[1], next_state[0]
                    action_value = reward + self.gamma * self.V[ny, nx]
                    action_values.append(action_value)
                
                self.policy[y, x] = np.argmax(action_values)
        
        self.trained = True
        return self.policy, self.errors

    def choose_action(self, state):
        y, x = state[1], state[0]
        return self.policy[y, x]

    def save_table(self, policy_file="policy_vi.npy", V_file="V_vi.npy"):
        np.save(policy_file, self.policy)
        np.save(V_file, self.V)

    def load_table(self, policy_file="policy_vi.npy", V_file="V_vi.npy"):
        self.policy = np.load(policy_file)
        self.V = np.load(V_file)
        self.trained = True