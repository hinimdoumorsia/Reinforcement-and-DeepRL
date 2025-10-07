import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.size, env.size, env.action_space.n))

    def choose_action(self, state):
        y, x = state[1], state[0]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.Q[y, x])

    def train(self, episodes=1000):
        errors = []
        policy = np.zeros((self.env.size, self.env.size))
        
        for _ in range(episodes):
            state = self.env.reset()
            done = False
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                y, x = state[1], state[0]
                ny, nx = next_state[1], next_state[0]
                
                self.Q[y, x, action] += self.alpha * (reward + self.gamma * np.max(self.Q[ny, nx]) - self.Q[y, x, action])
                state = next_state
            
            # Erreur / convergence
            current_policy = np.argmax(self.Q, axis=2)
            error = np.mean(np.abs(current_policy - policy))
            errors.append(error)
            policy = current_policy.copy()
        
        return policy, errors

    # ---- Sauvegarde / Chargement ----
    def save_table(self, filename="qlearning_Q.npy"):
        np.save(filename, self.Q)

    def load_table(self, filename="qlearning_Q.npy"):
        self.Q = np.load(filename)
