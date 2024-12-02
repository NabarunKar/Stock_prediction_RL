import gym
from gym import spaces
import pandas as pd
import numpy as np

class AdvancedStockEnv(gym.Env):
    def __init__(self, data, initial_balance=10000, transaction_cost=0.001, max_shares=1000):
        super(AdvancedStockEnv, self).__init__()
        self.data = data.reset_index(drop=True)
        self.data['Date'] = pd.to_datetime(self.data['Date']).astype(int) / 10**9
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.max_shares = max_shares

        self.current_step = 0
        self.balance = initial_balance
        self.shares_held = 0
        self.net_worth = initial_balance
        self.max_steps = len(data) - 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(len(data.columns) + 2,), dtype=np.float32
        )

        self.action_space = spaces.Discrete(3) 

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        self.shares_held = 0
        self.net_worth = self.initial_balance
        return self._get_observation()

    def step(self, action):
        current_price = self.data.iloc[self.current_step]['Close']
        reward = 0
        transaction_cost = 0

        if action == 1:  # Buy
            shares_bought = min(
                self.balance / current_price, self.max_shares - self.shares_held
            )
            transaction_cost = shares_bought * current_price * self.transaction_cost
            self.balance -= shares_bought * current_price + transaction_cost
            self.shares_held += shares_bought
        elif action == 2:  # Sell
            transaction_cost = self.shares_held * current_price * self.transaction_cost
            self.balance += self.shares_held * current_price - transaction_cost
            self.shares_held = 0

        self.net_worth = self.balance + self.shares_held * current_price
        reward = self.net_worth - self.initial_balance

        self.current_step += 1
        done = self.current_step >= self.max_steps

        return self._get_observation(), reward, done, {}

    def _get_observation(self):
      current_row = self.data.iloc[self.current_step]
      return np.array(list(current_row) + [self.balance, self.shares_held], dtype=np.float32)


    def render(self, mode="human"):
        print(f"Step: {self.current_step}, Net Worth: {self.net_worth}, Balance: {self.balance}, Shares Held: {self.shares_held}")

