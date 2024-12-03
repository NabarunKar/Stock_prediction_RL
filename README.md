# Stock Trading RL Agents
Here’s a well-structured and detailed README file for the stock trading simulation project using the S&P 500 stock dataset.

---

# **Stock Trading Simulation Using Reinforcement Learning**

This project demonstrates a stock trading simulation environment using reinforcement learning (RL). It implements various RL agents to learn and optimize trading strategies, providing insights into their performance in a realistic stock market setting. The dataset used in this simulation is based on S&P 500 stock data.

---

## **Table of Contents**

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Environment Details](#environment-details)
- [Implemented Agents](#implemented-agents)
- [Usage Instructions](#usage-instructions)
- [Results and Visualizations](#results-and-visualizations)
- [Future Work](#future-work)
- [License](#license)

---

## **Overview**

This project creates a customizable and modular environment for stock trading simulation, where agents interact with historical stock data to learn trading strategies.  
Key objectives include:  
- Exploring RL algorithms such as Q-Learning, SARSA, and Deep Q-Networks.  
- Evaluating trading performance based on balance and net worth.  
- Comparing RL-based strategies with baseline models.  

---

## **Features**

- **Trading Environment**:  
  A custom OpenAI Gym-style environment for stock trading simulation.  

- **Support for Multiple Agents**:  
  - Random Agent  
  - Tabular Q-Learning Agent  
  - Tabular SARSA Agent  
  - Deep Q-Network Agent (using PyTorch)  

- **Realistic Constraints**:  
  Includes transaction costs and maximum shares per transaction.  

- **Scalability**:  
  Environment and agents are easily extensible for additional features or advanced RL algorithms.  

---

## **Dataset**

The project uses the **S&P 500 stock dataset**, which contains daily stock prices (Open, High, Low, Close, Volume) for multiple companies.  
The dataset was preprocessed to include:  
- **Adjusted Close Prices**: To account for stock splits and dividends.  
- **Normalized Features**: For efficient training of RL agents.  
- **Timestamps**: Representing trading dates.

**Sample Data Format**:  
| Date       | Open  | High  | Low   | Close | Adj Close | Volume    |  
|------------|-------|-------|-------|-------|-----------|-----------|  
| 2023-01-01 | 100.0 | 105.0 | 98.0  | 102.0 | 102.0     | 1,000,000 |  

---

## **Environment Details**

### **State Space**:
The state includes the following:  
- Stock price information (`Open`, `High`, `Low`, `Close`, `Volume`).  
- Agent's current balance and held shares.  

### **Action Space**:  
Discrete actions:  
1. `Hold`  
2. `Buy`  
3. `Sell`  

### **Reward Function**:
The reward is the change in net worth relative to the initial balance, encouraging strategies that maximize portfolio value.  

---

## **Implemented Agents**

### 1. **Random Agent**:
Executes random actions as a baseline for comparison.  

### 2. **Q-Learning Agent**:
Uses tabular Q-Learning to update Q-values based on the agent's state-action pairs.  

### 3. **SARSA Agent**:
Learns on-policy by updating Q-values based on the current action and the subsequent state-action pair.  

### 4. **Deep Q-Network (DQN) Agent**:
Uses a neural network to approximate Q-values, incorporating:  
- Experience replay.  
- Epsilon-greedy exploration.  

---

## **Usage Instructions**

### **Prerequisites**:
- Python 3.8 or higher  
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `gym`, `torch`  

Install dependencies:  
```bash
pip install -r requirements.txt
```

### **Steps to Run**:

1. **Clone the Repository**:  
   ```bash
   git clone https://github.com/username/stock-trading-rl.git
   cd stock-trading-rl
   ```

2. **Prepare the Dataset**:  
   - Place the S&P 500 dataset CSV file in the `data/` directory.  
   - Preprocessing will normalize the data and split it into training and testing sets.

3. **Run the Simulation**:  
   Execute the training scripts for different agents:  
   - Random Agent:  
     ```bash
     python random_agent.py
     ```  
   - Q-Learning Agent:  
     ```bash
     python q_learning_agent.py
     ```  
   - SARSA Agent:  
     ```bash
     python sarsa_agent.py
     ```  
   - DQN Agent:  
     ```bash
     python dqn_agent.py
     ```  

---

## **Results and Visualizations**

The agents' performance is evaluated based on:  
1. Final Portfolio Value.  
2. Total Reward.  
3. Action Distributions.  

Sample visualization:  
- **Equity Curve**: Displays the portfolio's growth over time.  
- **Action Frequencies**: Highlights agent behavior (e.g., buy/sell frequency).

---

## **Future Work**

- Extend to continuous action spaces with advanced algorithms like PPO and DDPG.  
- Integrate more technical indicators in the observation space.  
- Add risk management features such as stop-loss and take-profit.  
- Implement a live trading bot using trained models.  

Summary :

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
