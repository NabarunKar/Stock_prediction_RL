import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from agent import RandomAgent, QLearningAgent, DQNAgent, SARSAAgent
from environment import AdvancedStockEnv 
data = pd.read_csv("^GSPC_2011(1).csv") 
env = AdvancedStockEnv(data) 
random_agent = RandomAgent(env.action_space.n)
q_learning_agent = QLearningAgent(env.observation_space.shape[0], env.action_space.n)
dqn_agent = DQNAgent(env.observation_space.shape[0], env.action_space.n)
sarsa_agent = SARSAAgent(env.observation_space.shape[0], env.action_space.n) 
def train_agent(env, agent, agent_type="random", episodes=100):
    rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        if agent_type == "sarsa":
            action = agent.act(state)
        while not done:
            if agent_type == "random":
                action = agent.act()
            elif agent_type != "sarsa":
                action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            if agent_type == "sarsa":
                next_action = agent.act(next_state)
                agent.learn(state, action, reward, next_state, next_action, done)
                action = next_action
            elif agent_type != "random":
                agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        rewards.append(total_reward)
        print(f"Episode {episode+1}/{episodes}, Total Reward: {total_reward}")
    return rewards

random_rewards = train_agent(env, random_agent, agent_type="random", episodes=10)
q_learning_rewards = train_agent(env, q_learning_agent, agent_type="q_learning", episodes=10)
dqn_rewards = train_agent(env, dqn_agent, agent_type="dqn", episodes=10)
sarsa_rewards = train_agent(env, sarsa_agent, agent_type="sarsa", episodes=10) 
plt.figure(figsize=(12, 6))
plt.plot(random_rewards, label='Random Agent')
plt.plot(q_learning_rewards, label='Q-Learning Agent')
plt.plot(dqn_rewards, label='DQN Agent')
plt.plot(sarsa_rewards, label='SARSA Agent')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Agent Performance Comparison')
plt.legend()
plt.show() 
agents = {
    "Random": np.mean(random_rewards),
    "Q-Learning": np.mean(q_learning_rewards),
    "DQN": np.mean(dqn_rewards),
    "SARSA": np.mean(sarsa_rewards)
}

best_agent = max(agents, key=agents.get)
print(f"The best performing agent is: {best_agent}")
