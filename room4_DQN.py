import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import deque
import json
import os

class Room4Env:
    def __init__(self, randomize_map=False):
        self.dt = 0.02
        self.room_size = 10.0
        self.goal_radius = 0.5

        if randomize_map:
            self.goal_pos = np.random.uniform(1.0, 9.0, size=2)
            self.start_pos = np.random.uniform(1.0, 9.0, size=2)
            # נוודא שהמרחק ההתחלתי מספיק גדול כדי לאפשר למידה משמעותית
            while np.linalg.norm(self.start_pos - self.goal_pos) < 4.0:
                self.start_pos = np.random.uniform(1.0, 9.0, size=2)
        else:
            self.goal_pos = np.array([9.5, 9.5])
            self.start_pos = np.array([0.5, 0.5])
            
        self.velocities = [-1, 0, 1]
        self.action_to_vel = []
        for vx in self.velocities:
            for vy in self.velocities:
                self.action_to_vel.append(np.array([vx, vy]))
                
        self.nA = len(self.action_to_vel)
        self.state = None

    def reset(self, exploring_starts=False):
        if exploring_starts:
            rand_x = np.random.uniform(1.0, self.room_size - 1.0)
            rand_y = np.random.uniform(1.0, self.room_size - 1.0)
            self.agent_pos = np.array([rand_x, rand_y], dtype=np.float32)
        else:
            self.agent_pos = np.copy(self.start_pos)
            
        self.agent_vel = np.zeros(2, dtype=np.float32)
        self.state = np.array([self.agent_pos[0], self.agent_pos[1], self.agent_vel[0], self.agent_vel[1]], dtype=np.float32)
        return self.state.copy()

    def step(self, action):
        x, y, _, _ = self.state
        vx, vy = self.action_to_vel[action]
        
        new_x = np.clip(x + vx * self.dt, 0.0, self.room_size)
        new_y = np.clip(y + vy * self.dt, 0.0, self.room_size)

        self.agent_pos = np.array([new_x, new_y], dtype=np.float32)
        
        self.state = np.array([new_x, new_y, vx, vy], dtype=np.float32)
        dist_to_goal = np.linalg.norm(np.array([new_x, new_y]) - self.goal_pos)
        done = dist_to_goal < self.goal_radius
        
        reward = 100.0 if done else -0.1 
        return self.state.copy(), reward, done, {}

class Room4OptimizedWrapper:
    def __init__(self, env):
        self.env = env
        self.nA = env.nA
        self.max_x = env.room_size 
        self.max_y = env.room_size
        self.max_v = 1.0 
        self.goal_x = env.goal_pos[0]
        self.goal_y = env.goal_pos[1]
        self.prev_dist = None  # הוספנו משתנה לשמירת המרחק הקודם

    def _normalize_state(self, state):
        x, y, vx, vy = state
        norm_x = x / self.max_x
        norm_y = y / self.max_y
        norm_vx = (vx + self.max_v) / (2 * self.max_v)
        norm_vy = (vy + self.max_v) / (2 * self.max_v)
        return np.array([norm_x, norm_y, norm_vx, norm_vy], dtype=np.float32)

    def reset(self, exploring_starts=False):
        state = self.env.reset(exploring_starts=exploring_starts)
        x, y, _, _ = state
        # חישוב ושמירת המרחק ההתחלתי
        self.prev_dist = np.sqrt((x - self.goal_x)**2 + (y - self.goal_y)**2)
        return self._normalize_state(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        x, y, _, _ = next_state
        
        current_dist = np.sqrt((x - self.goal_x)**2 + (y - self.goal_y)**2)
        
        # שיטת התגמול המבוססת על הפרש מרחקים (כמו בחדר 5)
        if not done:
            shaped_reward = (self.prev_dist - current_dist) * 10.0 
            reward += shaped_reward
        else:
            reward += 100.0 
            
        self.prev_dist = current_dist
        return self._normalize_state(next_state), reward, done, info

class QNetwork(nn.Module):
    def __init__(self, state_size=4, action_size=9, hidden_size=64):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size=10000, batch_size=64):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)

class DQNAgent:
    def __init__(self, env, learning_rate=1e-3, gamma=0.99, buffer_size=10000, batch_size=64, target_update_freq=100):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.q_network = QNetwork(state_size=4, action_size=env.nA)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        self.target_network = copy.deepcopy(self.q_network)
        self.target_network.eval() 
        
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)
        
    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.env.nA - 1)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()
            
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.memory.sample()
        q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def train_dqn(env, episodes=400, learning_rate=1e-3, gamma=0.99, batch_size=64, epsilon_decay=0.995, save_freq=100, progress_bar=None, status_text=None, live_callback=None, target_update_freq=100):
    agent = DQNAgent(env, learning_rate=learning_rate, gamma=gamma, batch_size=batch_size, target_update_freq=target_update_freq)
    epsilon = 1.0
    epsilon_min = 0.05
    rewards_history = []
    episode_steps = []
    epsilons = []
    checkpoints = {} 
    
    for ep in range(episodes):
        state = env.reset(exploring_starts=True)
        total_reward = 0
        steps_taken = 0
        done = False
        
        path_x, path_y = [], []
        curr_pos = env.env.state[:2]
        path_x.append(curr_pos[0])
        path_y.append(curr_pos[1])
        
        for step in range(1200): 
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            curr_pos = env.env.state[:2]
            path_x.append(curr_pos[0])
            path_y.append(curr_pos[1])
            
            agent.memory.add(state, action, reward, next_state, done)
            agent.train_step()
            
            state = next_state
            total_reward += reward
            steps_taken += 1

            if done: break
                
        if progress_bar is not None:
            progress_bar.progress(int(((ep + 1) / episodes) * 100))
        if status_text is not None:
            status_text.text(f"מתאמן... עברנו {ep + 1} מתוך {episodes} אפיזודות")
            
        if 'live_callback' in locals() and live_callback is not None and (ep + 1) % 20 == 0:
            live_callback(ep + 1, path_x, path_y, env.env, total_reward, epsilon)
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        rewards_history.append(total_reward)
        episode_steps.append(steps_taken)
        epsilons.append(epsilon)

        if (ep + 1) % save_freq == 0:
            checkpoints[ep + 1] = copy.deepcopy(agent.q_network.state_dict())

    if episodes not in checkpoints:
        checkpoints[episodes] = copy.deepcopy(
            agent.q_network.state_dict()
        )
            
    return agent, rewards_history, episode_steps, epsilons, checkpoints

# בלוק בדיקה
if __name__ == '__main__':
    base_env = Room4Env()
    optimized_env = Room4OptimizedWrapper(base_env)
    print("Testing Room 4 (Continuous DQN) logic for 10 episodes...")
    agent, rewards, steps, epsilons, checkpoints = train_dqn(optimized_env, episodes=10)
    print("DQN Run Complete!")
