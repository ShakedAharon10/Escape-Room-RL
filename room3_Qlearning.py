import numpy as np
import random

class Room3Env:
    def __init__(self, randomize_map=False):
        self.nrow = 10
        self.ncol = 10
        self.nS = self.nrow * self.ncol
        self.nA = 4
        
        if randomize_map:
            self._generate_random_layout()
        else:
            self.start_state = (0, 0)
            self.goal_state = (9, 9)
            self.walls = [(2, 2), (2, 3), (2, 4), (2, 5)] 
            self.cliff = [(8, col) for col in range(1, 9)]
            self.slippery_cells = [(5, 5), (6, 5), (7, 5)]
        
        self.current_state = self.start_state

    def _generate_random_layout(self):
        all_cells = [(r, c) for r in range(self.nrow) for c in range(self.ncol)]
        random.shuffle(all_cells)
        
        self.start_state = all_cells.pop()
        self.goal_state = all_cells.pop()
        
        self.walls = [all_cells.pop() for _ in range(4)]
        self.cliff = [all_cells.pop() for _ in range(8)] 
        self.slippery_cells = [all_cells.pop() for _ in range(3)]

    def _to_s(self, row, col):
        return row * self.ncol + col

    def reset(self, exploring_starts=False):
        """
        מאפס את הסביבה. 
        exploring_starts=True : מגריל נקודת התחלה (מומלץ לאימון)
        exploring_starts=False: מתחיל מנקודת ההתחלה הרשמית (מומלץ לתצוגה)
        """
        if exploring_starts:
            import random
            while True:
                r = random.randint(0, self.nrow - 1)
                c = random.randint(0, self.ncol - 1)
                
                # אנחנו מוודאים שהסוכן לא נוחת על קיר או על המטרה
                # בחדר 3 כדאי להוסיף לתנאי גם את ה-cliff כדי שלא ימות מיד
                if (r, c) not in self.walls and (r, c) != self.goal_state:
                    self.current_state = (r, c)
                    break
        else:
            self.current_state = self.start_state
            
        return self._to_s(*self.current_state)

    def step(self, action):
        row, col = self.current_state
        
        if (row, col) == self.goal_state:
            return self._to_s(*self.goal_state), 0.0, True, {}
            
        if (row, col) in self.slippery_cells:
            roll = random.random()
            if roll < 0.1: action = (action + 1) % 4
            elif roll < 0.2: action = (action - 1) % 4
                
        next_row, next_col = row, col
        if action == 0:   next_row = max(row - 1, 0)
        elif action == 1: next_col = min(col + 1, self.ncol - 1)
        elif action == 2: next_row = min(row + 1, self.nrow - 1)
        elif action == 3: next_col = max(col - 1, 0)
        
        if (next_row, next_col) in self.walls:
            next_row, next_col = row, col
            
        self.current_state = (next_row, next_col)
        next_s = self._to_s(next_row, next_col)
        
        done = False
        if (next_row, next_col) == self.goal_state:
            reward = 100.0
            done = True
        elif (next_row, next_col) in self.cliff:
            reward = -100.0 
            done = True     
        else:
            reward = -1.0
            
        return next_s, reward, done, {}

def epsilon_greedy_policy(Q, state, epsilon, nA):
    if np.random.rand() < epsilon:
        return np.random.randint(nA)
    else:
        q_values = Q[state]
        max_val = np.max(q_values)
        best_actions = np.where(np.isclose(q_values, max_val))[0]
        return np.random.choice(best_actions)

def q_learning(env, num_episodes=2000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, save_freq=500, live_callback=None):
    Q = np.zeros((env.nS, env.nA))
    episode_rewards = []
    episode_steps = []
    epsilons = []
    epsilon = epsilon_start
    checkpoints = {} 

    for episode in range(num_episodes):
        state = env.reset(exploring_starts=True)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon, env.nA)
            next_state, reward, done, _ = env.step(action)
            
            target = reward + gamma * np.max(Q[next_state])
            
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            
            state = next_state
            total_reward += reward
            steps += 1
            
        episode_rewards.append(total_reward)
        episode_steps.append(steps)
        epsilons.append(epsilon)
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if (episode + 1) % save_freq == 0:
            checkpoints[episode + 1] = Q.copy()
            if live_callback:
                live_callback(episode + 1, Q, env)

    if num_episodes not in checkpoints:
        checkpoints[num_episodes] = Q.copy()
        
    return Q, episode_rewards, episode_steps, epsilons, checkpoints

def get_policy_grid_string(env, policy):
    action_symbols = {0: '↑', 1: '→', 2: '↓', 3: '←'}
    grid = np.empty((env.nrow, env.ncol), dtype=object)
    
    for row in range(env.nrow):
        for col in range(env.ncol):
            s = row * env.ncol + col
            if (row, col) == env.goal_state:
                grid[row, col] = ' G '
            elif (row, col) in env.walls:
                grid[row, col] = ' W '
            else:
                best_a = np.argmax(policy[s])
                symbol = action_symbols[best_a]
                if (row, col) in env.slippery_cells:
                    grid[row, col] = f"~{symbol}~"
                else:
                    grid[row, col] = f" {symbol} "
                    
    output = "--- Policy Map ---\n"
    for row in range(env.nrow):
        output += " | ".join([str(cell).ljust(3) for cell in grid[row]]) + "\n"
        output += "-" * (env.ncol * 6) + "\n"
    return output

# בלוק בדיקה
if __name__ == '__main__':
    env = Room3Env()
    print("Training Q-Learning for 100 episodes...")
    Q, rewards, steps, epsilons, checkpoints = q_learning(env, num_episodes=100)
    
    learned_policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        best_a = np.argmax(Q[s])
        learned_policy[s, best_a] = 1.0
        
    print(get_policy_grid_string(env, learned_policy))
