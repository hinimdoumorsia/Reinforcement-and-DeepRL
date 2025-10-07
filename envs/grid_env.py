import gym
from gym import spaces
import numpy as np
import pygame

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, size=6, start_pos=(0,0), goal_positions=[(5,5)],
                 obstacles=[(1,1),(2,2),(3,1)], max_steps=50, cell_size=60):
        super(GridWorld, self).__init__()
        self.size = size
        self.start_pos = np.array(start_pos)
        self.goal_positions = [np.array(pos) for pos in goal_positions]
        self.obstacles = [np.array(obs) for obs in obstacles]
        self.max_steps = max_steps
        self.cell_size = cell_size
        self.agent_pos = self.start_pos.copy()
        self.steps = 0

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiDiscrete([size, size])

        # Pygame setup
        pygame.init()
        self.window_size = self.size * self.cell_size
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("GridWorld")
        self.clock = pygame.time.Clock()

    def reset(self):
        self.agent_pos = self.start_pos.copy()
        self.steps = 0
        return tuple(self.agent_pos)

    def step(self, action):
        self.steps += 1
        old_pos = self.agent_pos.copy()
        
        # Actions: 0=haut, 1=droite, 2=bas, 3=gauche
        if action == 0 and self.agent_pos[1] < self.size-1:  # haut
            self.agent_pos[1] += 1
        elif action == 1 and self.agent_pos[0] < self.size-1:  # droite
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:  # bas
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[0] > 0:  # gauche
            self.agent_pos[0] -= 1

        # Collision avec obstacle
        if any((self.agent_pos == obs).all() for obs in self.obstacles):
            self.agent_pos = old_pos
            reward = -1
            done = False
        # Atteindre un goal
        elif any((self.agent_pos == goal).all() for goal in self.goal_positions):
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        done = done or self.steps >= self.max_steps
        return tuple(self.agent_pos), reward, done, {}

    def simulate_step(self, state, action):
        """Méthode pour la simulation sans changer l'état réel de l'environnement"""
        x, y = state
        new_x, new_y = x, y
        
        # Actions: 0=haut, 1=droite, 2=bas, 3=gauche
        if action == 0 and y < self.size-1:  # haut
            new_y += 1
        elif action == 1 and x < self.size-1:  # droite
            new_x += 1
        elif action == 2 and y > 0:  # bas
            new_y -= 1
        elif action == 3 and x > 0:  # gauche
            new_x -= 1

        new_state = (new_x, new_y)
        
        # Vérifier collision avec obstacle
        if any((np.array(new_state) == obs).all() for obs in self.obstacles):
            new_state = state  # Reste sur place
            reward = -1
            done = False
        # Vérifier si goal atteint
        elif any((np.array(new_state) == goal).all() for goal in self.goal_positions):
            reward = 10
            done = True
        else:
            reward = -0.1
            done = False

        return new_state, reward, done, {}

    def render(self, mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        self.screen.fill((255, 255, 255))  # fond blanc

        # Dessiner obstacles
        for obs in self.obstacles:
            pygame.draw.rect(
                self.screen, (0, 0, 0),
                (obs[0]*self.cell_size, (self.size-1-obs[1])*self.cell_size, self.cell_size, self.cell_size)
            )

        # Dessiner agent
        pygame.draw.rect(
            self.screen, (0, 0, 255),
            (self.agent_pos[0]*self.cell_size, (self.size-1-self.agent_pos[1])*self.cell_size, self.cell_size, self.cell_size)
        )

        # Dessiner objectifs
        for goal in self.goal_positions:
            pygame.draw.rect(
                self.screen, (0, 255, 0),
                (goal[0]*self.cell_size, (self.size-1-goal[1])*self.cell_size, self.cell_size, self.cell_size)
            )

        # Grille
        for x in range(self.size+1):
            pygame.draw.line(self.screen, (0,0,0), (x*self.cell_size,0), (x*self.cell_size,self.window_size))
        for y in range(self.size+1):
            pygame.draw.line(self.screen, (0,0,0), (0,y*self.cell_size), (self.window_size,y*self.cell_size))

        pygame.display.flip()
        self.clock.tick(5)