import numpy as np
import random

class Room1Env:
    def __init__(self, randomize_map=False):
    
        self.nrow = 10
        self.ncol = 10
        self.nS = self.nrow * self.ncol  # 100 מצבים סה"כ
        self.nA = 4                     # 4 פעולות: 0=Up, 1=Right, 2=Down, 3=Left
        
        # כאן אנחנו בודקים האם ה-Checkbox סומן
        if randomize_map:
            self._generate_random_layout()
        else:
            self.start_state = (0, 0)
            self.goal_state = (9, 9)
            self.slippery_cells = [(2, 2), (2, 3), (5, 5), (5, 6), (7, 8)]
            self.walls = [(3, 3), (3, 4), (3, 5), (7, 2), (7, 3)] 
        
        self.P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        self._build_transitions()


    def _generate_random_layout(self):
        # יצירת רשימה של כל התאים האפשריים
        all_cells = [(r, c) for r in range(self.nrow) for c in range(self.ncol)]
        random.shuffle(all_cells)
        
        # שליפת מיקומים ייחודיים ללא כפילויות
        self.start_state = all_cells.pop()
        self.goal_state = all_cells.pop()
        
        self.walls = [all_cells.pop() for _ in range(5)]
        self.slippery_cells = [all_cells.pop() for _ in range(5)]

    def _to_s(self, row, col):
        return row * self.ncol + col

    def reset(self):
        return self._to_s(*self.start_state)

    def _inc(self, row, col, action):
        if action == 0:   # Up
            row = max(row - 1, 0)
        elif action == 1: # Right
            col = min(col + 1, self.ncol - 1)
        elif action == 2: # Down
            row = min(row + 1, self.nrow - 1)
        elif action == 3: # Left
            col = max(col - 1, 0)
        return (row, col)

    def _build_transitions(self):
        for row in range(self.nrow):
            for col in range(self.ncol):
                s = self._to_s(row, col)

                if (row, col) == self.goal_state:
                    for a in range(self.nA):
                        self.P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                
                if (row, col) in self.walls:
                    for a in range(self.nA):
                        self.P[s][a] = [(1.0, s, 0.0, True)]
                    continue

                is_slippery = (row, col) in self.slippery_cells

                for a in range(self.nA):
                    if is_slippery:
                        transitions = [
                            (0.8, a),
                            (0.1, (a + 1) % 4),
                            (0.1, (a - 1) % 4)
                        ]
                    else:
                        transitions = [(1.0, a)]

                    for prob, move_action in transitions:
                        next_row, next_col = self._inc(row, col, move_action)
                        if (next_row, next_col) in self.walls:
                            next_row, next_col = row, col
                            
                        next_s = self._to_s(next_row, next_col)
                        done = (next_row, next_col) == self.goal_state
                        reward = 100.0 if done else -1.0
                        self.P[s][a].append((prob, next_s, reward, done))

def policy_evaluation(policy, env, discount_factor=0.99, theta=0.0001):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            v = 0
            for a, action_prob in enumerate(policy[s]):
                for prob, next_s, reward, done in env.P[s][a]:
                    v += action_prob * prob * (reward + discount_factor * V[next_s])
            delta = max(delta, np.abs(v - V[s]))
            V[s] = v
        if delta < theta:
            break
    return np.array(V)

def policy_iteration(env, discount_factor=0.99, theta=0.0001):
    policy = np.ones([env.nS, env.nA]) / env.nA
    policy_changes_history = []

    while True:
        V = policy_evaluation(policy, env, discount_factor, theta)
        policy_stable = True
        changes_this_iter = 0

        for s in range(env.nS):
            chosen_a = np.argmax(policy[s])
            action_values = np.zeros(env.nA)

            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_s])

            best_a = np.argmax(action_values)

            if chosen_a != best_a:
                policy_stable = False
                changes_this_iter += 1

            policy[s] = np.eye(env.nA)[best_a]

        policy_changes_history.append(changes_this_iter)

        if policy_stable:
            return policy, V, policy_changes_history

def value_iteration(env, discount_factor=0.99, theta=0.0001):
    V = np.zeros(env.nS)
    deltas_history = [] 

    while True:
        delta = 0
        for s in range(env.nS):
            v_old = V[s]
            action_values = np.zeros(env.nA)
            
            for a in range(env.nA):
                for prob, next_s, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_s])
                    
            V[s] = np.max(action_values)
            delta = max(delta, np.abs(v_old - V[s]))
            
        deltas_history.append(delta)
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_s, reward, done in env.P[s][a]:
                action_values[a] += prob * (reward + discount_factor * V[next_s])
        best_a = np.argmax(action_values)
        policy[s, best_a] = 1.0

    return policy, V, deltas_history

def get_policy_grid_string(env, policy):
    """
    מחזירה את הגריד כמחרוזת טקסט מעוצבת כדי שיהיה קל להציג ב-Streamlit
    """
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

# בלוק בדיקה - רץ רק אם מריצים את הקובץ ישירות
if __name__ == '__main__':
    env = Room1Env()
    optimal_policy, optimal_V, changes_history = policy_iteration(env)
    print(get_policy_grid_string(env, optimal_policy))