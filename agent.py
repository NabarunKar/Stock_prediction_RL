import random
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F 

class RandomAgent:
    def __init__(self, action_size):
        self.action_size = action_size

    def act(self):
        return random.choice([0, 1, 2])   

class QLearningAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def act(self, state):
        state_tuple = tuple(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))  
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_tuple])   

    def learn(self, state, action, reward, next_state, done):
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)

        next_state_tuple = tuple(next_state)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.action_size)

        target = reward + (self.gamma * np.max(self.q_table[next_state_tuple]) if not done else reward)
        self.q_table[state_tuple][action] += self.alpha * (target - self.q_table[state_tuple][action])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.memory = []
        self.batch_size = 32
        self.model = self._build_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def _build_model(self):
        return nn.Sequential(
            nn.Linear(self.state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_size),
        )
    def learn(self, state, action, reward, next_state, done):
        self.remember(state, action, reward, next_state, done)
        if len(self.memory) >= self.batch_size:
            self.replay()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item() 
    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target += self.gamma * np.max(self.model(torch.FloatTensor(next_state).unsqueeze(0)).detach().numpy())
            
            state = torch.FloatTensor(state).unsqueeze(0)
            target_f = self.model(state).detach().clone()
            target_f[0][action] = target

            output = self.model(state)
            loss = self.criterion(output, target_f)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
# SARSA Agent
class SARSAAgent:
    def __init__(self, state_size, action_size, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = {}

    def act(self, state):
        state_tuple = tuple(state)
        if np.random.rand() <= self.epsilon:
            return random.choice(range(self.action_size))
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        return np.argmax(self.q_table[state_tuple])

    def learn(self, state, action, reward, next_state, next_action, done):
        state_tuple = tuple(state)
        if state_tuple not in self.q_table:
            self.q_table[state_tuple] = np.zeros(self.action_size)
        next_state_tuple = tuple(next_state)
        if next_state_tuple not in self.q_table:
            self.q_table[next_state_tuple] = np.zeros(self.action_size)
        
        target = reward + (self.gamma * self.q_table[next_state_tuple][next_action] if not done else reward)
        self.q_table[state_tuple][action] += self.alpha * (target - self.q_table[state_tuple][action])
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay