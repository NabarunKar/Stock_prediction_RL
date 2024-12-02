# Stock Trading RL Agents

This project implements various reinforcement learning (RL) agents to solve the problem of stock trading. The agents include:

- **Random Agent**: Chooses actions randomly.
- **Q-Learning Agent**: Uses Q-learning algorithm to learn stock trading policies.
- **DQN (Deep Q-Network) Agent**: Implements deep Q-learning using a neural network.
- **SARSA Agent**: Uses SARSA (State-Action-Reward-State-Action) algorithm for learning.

The environment is built using OpenAI's `gym` library to simulate a stock market where the agents can interact, make decisions, and learn over time.

## Project Structure

- `agent.py`: Contains the definitions of the agents.
- `environment.py`: Contains the definition of the stock trading environment using `gym`.
- `train.py`: The main script to train and evaluate the agents.
- `README.md`: Project documentation.

## Requirements

- Python 3.x
- gym
- pandas
- numpy
- torch (for DQN)
- matplotlib (for plotting results)

You can install the dependencies using:

```bash
pip install gym pandas numpy torch matplotlib
