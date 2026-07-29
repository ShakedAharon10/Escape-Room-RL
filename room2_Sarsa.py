import numpy as np
import random

class Room2Env:
    def __init__(self, randomize_map=False):
        self.nrow = 10
        self.ncol = 10
        self.nS = self.nrow * self.ncol
        self.nA = 4 # 0:Up, 1:Right, 2:Down, 3:Left

        if randomize_map:
            self._generate_random_layout()
        else:
            self.start_state = (0, 0)
            self.goal_state = (9, 9)
            self.slippery_cells = [(2, 2), (2, 3), (5, 5), (5, 6), (7, 8)]
            self.walls = [(3, 3), (3, 4), (3, 5), (7, 2), (7, 3)] 
            self.traps = [(1, 1), (8, 8), (4, 4)] # מלכודות
        
        self.current_state = self.start_state

    def _generate_random_layout(self):
        all_cells = [(r, c) for r in range(self.nrow) for c in range(self.ncol)]
        random.shuffle(all_cells)
        
        # שליפת מיקומים ייחודיים ללא כפילויות
        self.start_state = all_cells.pop()
        self.goal_state = all_cells.pop()
        
        self.walls = [all_cells.pop() for _ in range(5)]
        self.traps = [all_cells.pop() for _ in range(3)]
        self.slippery_cells = [all_cells.pop() for _ in range(5)]

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
            if roll < 0.1:
                action = (action + 1) % 4
            elif roll < 0.2:
                action = (action - 1) % 4
                
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
        elif (next_row, next_col) in self.traps:
            reward = -50.0 
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
        best_actions = np.where(q_values == max_val)[0]
        return np.random.choice(best_actions)

def sarsa(env, num_episodes=2000, alpha=0.1, gamma=0.99, epsilon_start=1.0, epsilon_decay=0.995, epsilon_min=0.01, save_freq=500, live_callback=None):
    Q = np.zeros((env.nS, env.nA))
    episode_rewards = []
    episode_steps = []
    epsilons = []
    epsilon = epsilon_start
    checkpoints = {}

    for episode in range(num_episodes):
        state = env.reset(exploring_starts=True)
        action = epsilon_greedy_policy(Q, state, epsilon, env.nA)
        total_reward = 0
        steps = 0
        done = False
        
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon, env.nA)
            
            target = reward + gamma * Q[next_state][next_action]
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])
            
            state = next_state
            action = next_action
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

# בלוק בדיקה
if __name__ == '__main__':
    env = Room2Env()
    print("Training SARSA...")
    Q, rewards, steps, epsilons, checkpoints = sarsa(env, num_episodes=100)
    print(f"Training complete! Final reward: {rewards[-1]}")