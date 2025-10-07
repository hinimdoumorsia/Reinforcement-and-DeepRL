import numpy as np

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((env.size, env.size, env.action_space.n))
        self.episode_errors = []

    def choose_action(self, state):
        y, x = state[1], state[0]
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.env.action_space.n)
        return np.argmax(self.Q[y, x])

    def train(self, episodes=1000):
        self.episode_errors = []
        
        for episode in range(episodes):
            state = self.env.reset()
            done = False
            total_error = 0
            steps = 0
            
            while not done and steps < 100:  # Limite de pas
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                
                y, x = state[1], state[0]
                ny, nx = next_state[1], next_state[0]
                
                # Q-learning update
                old_q = self.Q[y, x, action]
                td_target = reward + self.gamma * np.max(self.Q[ny, nx])
                self.Q[y, x, action] += self.alpha * (td_target - old_q)
                
                # Calcul de l'erreur
                error = abs(td_target - old_q)
                total_error += error
                steps += 1
                
                state = next_state
            
            avg_error = total_error / max(steps, 1)
            self.episode_errors.append(avg_error)
        
        policy = np.argmax(self.Q, axis=2)
        return policy, self.episode_errors

    def save_table(self, filename="qlearning_Q.npy"):
        np.save(filename, self.Q)

    def load_table(self, filename="qlearning_Q.npy"):
        self.Q = np.load(filename)