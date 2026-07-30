# Escape Room - Deep Reinforcement Learning Simulation

An interactive educational simulation built with Python and Streamlit, demonstrating various Reinforcement Learning algorithms across 5 distinct "escape rooms"—ranging from classic dynamic programming to advanced deep Q-networks with dynamic obstacles and vision systems.

---

## 🏛️ Rooms Overview & Specifications

### Room 1 - Dynamic Programming (DP)
* **Core Algorithm:** Policy Iteration (Known Model)
* **State Space:** 10x10 Grid (100 discrete states)
* **Action Space:** 4 directions (up, down, right, left)
* **Unique Challenge:** Slippery cells creating stochastic uncertainty in movement.
* **Reward Structure:** Step penalty to encourage efficiency, positive terminal reward for reaching the goal, and heavy penalties for hazards.
* **Optimal Parameters:** $\gamma = 0.99$, Convergence Threshold ($\theta$) = 0.0001.

### Room 2 - SARSA (On-Policy)
* **Core Algorithm:** SARSA
* **State Space:** 10x10 Grid (100 discrete states)
* **Action Space:** 4 directions
* **Unique Challenge:** Unknown environment model, heavy traps (penalty -50) combined with slippery cells.
* **Reward Structure:** Sparse/dense reward shaping balancing path length and hazard avoidance under an on-policy evaluation framework.
* **Optimal Parameters:** Episodes = 1000, Learning Rate ($\alpha$) = 0.1, Discount Factor ($\gamma$) = 0.99, Epsilon Decay = 0.995.

### Room 3 - Q-Learning (Off-Policy)
* **Core Algorithm:** Q-Learning
* **State Space:** 10x10 Grid (100 discrete states)
* **Action Space:** 4 directions
* **Unique Challenge:** Dangerous cliff (penalty -100) creating a dilemma between a shortcut and a safe detour.
* **Reward Structure:** High negative feedback for falling off the cliff, encouraging the off-policy learner to find the optimal greedy path while exploring via $\epsilon$-greedy.
* **Optimal Parameters:** Episodes = 1000, Learning Rate ($\alpha$) = 0.1, Discount Factor ($\gamma$) = 0.99, Epsilon Decay = 0.995.

### Room 4 - Deep Q-Network in Continuous Space
* **Core Algorithm:** Deep Q-Network (DQN)
* **State Space:** Continuous: 10x10 meters, Position (X,Y) and Velocity (Vx,Vy)
* **Action Space:** 9 velocity combinations (discretized actions)
* **Unique Challenge:** Value function approximation in infinite space, neural networks, and Reward Shaping.
* **Reward Structure:** Continuous distance-based shaping rewards moving closer to the target, combined with collision penalties and velocity regularization.
* **Optimal Parameters:** Episodes = 800, Learning Rate ($\alpha$) = 0.001, Discount Factor ($\gamma$) = 0.99, Batch Size = 64, Epsilon Decay = 0.992, Target Update Frequency = 100.

### Room 5 - Double DQN with Vision and Dynamic Obstacles
* **Core Algorithm:** Double DQN
* **State Space:** 20-dimensional vector (X, Y, Vx, Vy + 16 Raycast sensors)
* **Action Space:** 9 velocity combinations
* **Unique Challenge:** Dodge dynamically generated obstacles and prevent overestimation using DDQN.
* **Reward Structure:** Dense rewards based on forward progress, penalty for proximity/collision with moving obstacles, and a large positive reward for reaching the goal.
* **Optimal Parameters:** Episodes = 1500, Learning Rate ($\alpha$) = 0.0005, Discount Factor ($\gamma$) = 0.99, Batch Size = 64, Epsilon Decay = 0.997, Target Update Frequency = 200, Checkpoint Frequency = every 50 episodes.

---

## 🚀 Installation & Running Locally

1. **Clone, install, and run:**
   ```bash
   git clone [https://github.com/ShakedAharon10/Escape-Room-RL.git](https://github.com/ShakedAharon10/Escape-Room-RL.git)
   cd Escape-Room-RL
   pip install -r requirements.txt
   streamlit run app.py

🌐 Live Demo: [Streamlit App](https://escape-room-rl-shaked-shiran-ronen.streamlit.app/)
