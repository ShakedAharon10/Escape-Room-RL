import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import copy
from collections import deque

class Room5Env:
    def __init__(self, num_obstacles=(5, 10), ray_length=3.0, randomize_map=False):
        self.dt = 0.02
        self.room_size = 10.0
        self.goal_radius = 0.5

        if randomize_map:
            self.goal_pos = np.random.uniform(1.0, 9.0, size=2)
            self.start_pos = np.random.uniform(1.0, 9.0, size=2)
            # הבטחת מרחק מינימלי בין התחלה לסיום
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
        
        self.num_obstacles_range = num_obstacles
        self.obstacle_radius = 0.25 
        self.ray_length = ray_length 
        self.num_rays = 16 
        
        self.state = None
        self.obstacles = []


    def reset(self, exploring_starts=False):
        if exploring_starts:
        
            rand_x = np.random.uniform(1.0, self.room_size - 1.0)
            rand_y = np.random.uniform(1.0, self.room_size - 1.0)
            self.agent_pos = np.array([rand_x, rand_y], dtype=np.float32)

        else:
            self.agent_pos = self.start_pos.copy().astype(np.float32)
            
        self.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
        
        num_obs = np.random.randint(self.num_obstacles_range[0], self.num_obstacles_range[1] + 1)
        self.obstacles = []
        while len(self.obstacles) < num_obs:
            obs_pos = np.random.uniform(1.0, 9.0, size=2)
            if np.linalg.norm(obs_pos - self.agent_pos) > 1.5 and np.linalg.norm(obs_pos - self.goal_pos) > 1.5:
                self.obstacles.append(obs_pos)
                
        return self._get_obs()
        
        num_obs = np.random.randint(self.num_obstacles_range[0], self.num_obstacles_range[1] + 1)
        self.obstacles = []
        while len(self.obstacles) < num_obs:
            obs_pos = np.random.uniform(1.0, 9.0, size=2)
            if np.linalg.norm(obs_pos - self.agent_pos) > 1.5 and np.linalg.norm(obs_pos - self.goal_pos) > 1.5:
                self.obstacles.append(obs_pos)
                
        return self._get_obs()

    def _get_obs(self):
        sector_dists = np.full(self.num_rays, self.ray_length)
        sector_angle = (2 * np.pi) / self.num_rays
        
        for obs in self.obstacles:
            v = obs - self.agent_pos
            dist = np.linalg.norm(v)
            if dist <= self.ray_length:
                angle = np.arctan2(v[1], v[0])
                if angle < 0: angle += 2 * np.pi
                sector = int(((angle + sector_angle/2) % (2*np.pi)) / sector_angle)
                if dist < sector_dists[sector]:
                    sector_dists[sector] = dist
                    
        return np.concatenate([self.agent_pos, self.agent_vel, sector_dists]).astype(np.float32)

    def step(self, action):
        vx, vy = self.action_to_vel[action]
        self.agent_vel = np.array([vx, vy], dtype=np.float32)
        
        self.agent_pos += self.agent_vel * self.dt
        self.agent_pos = np.clip(self.agent_pos, 0.0, self.room_size)
        
        obs = self._get_obs()
        dist_to_goal = np.linalg.norm(self.agent_pos - self.goal_pos)
        done = dist_to_goal < self.goal_radius
        
        hit_obstacle = False
        for obs_pos in self.obstacles:
            if np.linalg.norm(self.agent_pos - obs_pos) < self.obstacle_radius:
                hit_obstacle = True
                break
                
        if done: reward = 100.0
        elif hit_obstacle:
            reward = -20.0
            done = True
        else: reward = -0.1
            
        return obs, reward, done, {}

class Room5OptimizedWrapper:
    def __init__(self, env):
        self.env = env
        self.nA = env.nA
        self.max_x = env.room_size 
        self.max_y = env.room_size
        self.max_v = 1.0 
        self.goal_pos = env.goal_pos
        self.ray_length = env.ray_length
        self.prev_dist = None

    def _normalize_state(self, state):
        pos_x, pos_y = state[0]/self.max_x, state[1]/self.max_y
        vel_x, vel_y = (state[2]+self.max_v)/(2*self.max_v), (state[3]+self.max_v)/(2*self.max_v)
        rays = state[4:] / self.ray_length 
        return np.array([pos_x, pos_y, vel_x, vel_y] + rays.tolist(), dtype=np.float32)

    def reset(self, exploring_starts=False):
        state = self.env.reset(exploring_starts=exploring_starts)
        self.prev_dist = np.linalg.norm(self.env.agent_pos - self.goal_pos)
        return self._normalize_state(state)

    def step(self, action):
        next_state, base_reward, done, info = self.env.step(action)
        current_dist = np.linalg.norm(self.env.agent_pos - self.goal_pos)
        min_obstacle_dist = np.min(next_state[4:]) 
        
        if not done and base_reward != -20.0:
            shaped_reward = (self.prev_dist - current_dist) * 10.0 
            proximity_penalty = 0.0
            if min_obstacle_dist < 0.8:
                proximity_penalty = -2.0 * (0.8 - min_obstacle_dist)
            reward = base_reward + shaped_reward + proximity_penalty
        else:
            reward = base_reward
            
        self.prev_dist = current_dist
        return self._normalize_state(next_state), reward, done, info

class QNetworkRoom5(nn.Module):
    def __init__(self, state_size=20, action_size=9, hidden_size=128):
        super(QNetworkRoom5, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class ReplayBuffer:
    def __init__(self, buffer_size=100000, batch_size=64):
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

class DDQNAgentRoom5:
    def __init__(self, env, learning_rate=5e-4, gamma=0.99, buffer_size=100000, batch_size=64, target_update_freq=200):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps_done = 0
        
        self.q_network = QNetworkRoom5(state_size=20, action_size=env.nA)
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
            best_next_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
            max_next_q_values = self.target_network(next_states).gather(1, best_next_actions)
            
        targets = rewards + self.gamma * max_next_q_values * (1 - dones)
        loss = self.criterion(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

def train_room5(env, episodes=2000, learning_rate=5e-4, gamma=0.99, batch_size=64, epsilon_decay=0.995, save_freq=500, progress_bar=None, status_text=None, live_callback=None):
    agent = DDQNAgentRoom5(env, learning_rate=learning_rate, gamma=gamma, batch_size=batch_size)
    epsilon = 1.0
    epsilon_min = 0.01
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
        curr_pos = env.env.agent_pos 
        path_x.append(curr_pos[0])
        path_y.append(curr_pos[1])
        
        for step in range(1200): 
            action = agent.select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            
            curr_pos = env.env.agent_pos 
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
            status_text.text(f"Double DQN מתאמן במרחב... {ep + 1}/{episodes}")
            
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
    base_env5 = Room5Env(num_obstacles=(5, 10), ray_length=3.0)
    optimized_env5 = Room5OptimizedWrapper(base_env5)
    print("Testing Room 5 (DDQN) logic for 10 episodes...")
    agent, rewards, steps, epsilons, checkpoints = train_room5(optimized_env5, episodes=10)
    print("DDQN Run Complete!")