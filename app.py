import streamlit as st
import numpy as np
import time
import copy
import plotly.graph_objects as go
import plotly.express as px
import torch
import random

import room1_DP
import room2_Sarsa
import room3_Qlearning
import room4_DQN
import room5_DoubleDQN

def set_global_seed(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if 'seed_set' not in st.session_state:
    set_global_seed(42)
    st.session_state.seed_set = True

# ==========================================
# הגדרות עמוד ראשי -CSS חללי/סייברפאנק
# ==========================================
st.set_page_config(page_title="Hezki Escape Room", page_icon="🐶", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0b0f19; color: #e2e8f0; }
    [data-testid="stSidebar"] { background-color: #111827 !important; border-right: 1px solid #1f2937; }
    h1, h2, h3, h4 { font-family: 'Trebuchet MS', sans-serif; color: #00f3ff !important; text-shadow: 0 0 10px rgba(0, 243, 255, 0.5); }
    .metric-container { display: flex; justify-content: space-between; gap: 15px; margin-bottom: 20px; }
    .metric-card { background: linear-gradient(145deg, #1f2937, #111827); border: 1px solid #00f3ff; padding: 20px; border-radius: 12px; text-align: center; box-shadow: 0 0 15px rgba(0, 243, 255, 0.2); flex: 1; transition: transform 0.3s; }
    .metric-card:hover { transform: translateY(-5px); box-shadow: 0 0 25px rgba(0, 243, 255, 0.5); }
    .metric-card h4 { margin: 0; font-size: 16px; color: #9ca3af !important; text-shadow: none; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { margin: 10px 0 0 0; font-size: 32px; color: #39ff14 !important; text-shadow: 0 0 10px rgba(57, 255, 20, 0.4); }
    .stButton>button { background: linear-gradient(90deg, #ff00ff, #8a2be2); color: white !important; border-radius: 8px; font-weight: 800; font-size: 16px; border: none; padding: 10px 24px; box-shadow: 0 0 15px rgba(255, 0, 255, 0.4); transition: all 0.3s ease; width: 100%; }
    .stButton>button:hover { transform: scale(1.02); box-shadow: 0 0 25px rgba(255, 0, 255, 0.8); background: linear-gradient(90deg, #8a2be2, #ff00ff); }
    .grid-cell { display: flex; align-items: center; justify-content: center; font-size: 24px; border-radius: 8px; transition: all 0.2s; }
    .info-box { background-color: rgba(0, 243, 255, 0.05); border-left: 5px solid #00f3ff; padding: 15px; margin-bottom: 20px; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# פונקציות עזר לתצוגה, גרפים ואקדמיה
# ==========================================

def show_room_info(algo_name, state_space, action_space, challenge):
    st.markdown(f"""
    <div class="info-box">
        <h4>📋 תעודת זהות - סביבת למידה</h4>
        <ul style="list-style-type: none; padding: 0; margin: 0; color: #e2e8f0; font-size: 16px;">
            <li><strong>🧠 Core Algorithm:</strong> {algo_name}</li>
            <li><strong>🌍 State Space:</strong> {state_space}</li>
            <li><strong>🕹️ Action Space:</strong> {action_space}</li>
            <li><strong>⚠️ Unique Challenge:</strong> {challenge}</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def render_live_metrics(ep, total_episodes, reward, epsilon):
    html = f"""
    <div class="metric-container">
        <div class="metric-card"><h4>📍 אפיזודה</h4><h2>{ep} / {total_episodes}</h2></div>
        <div class="metric-card"><h4>💰 תגמול מצטבר</h4><h2>{reward:.2f}</h2></div>
        <div class="metric-card"><h4>🔍 קצב חקירה (Epsilon)</h4><h2>{epsilon:.3f}</h2></div>
    </div>
    """
    return html

def plot_training_metrics(rewards, steps, epsilons, window=50):
    tab1, tab2, tab3 = st.tabs(["💰 תגמול ממוצע", "👟 צעדים לפרק", "📉 חקירה (Epsilon)"])
    smoothed_rewards = [np.mean(rewards[max(0, i-window):i+1]) for i in range(len(rewards))]
    smoothed_steps = [np.mean(steps[max(0, i-window):i+1]) for i in range(len(steps))]
    
    layout_dark = dict(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), xaxis=dict(showgrid=True, gridcolor='#1f2937'), yaxis=dict(showgrid=True, gridcolor='#1f2937'))
    
    with tab1:
        fig = go.Figure(go.Scatter(y=smoothed_rewards, mode='lines', line=dict(color='#39ff14', width=3)))
        fig.update_layout(title="Average Reward per Episode", xaxis_title="Episodes", yaxis_title="Reward", **layout_dark)
        st.plotly_chart(fig, use_container_width=True)
    with tab2:
        fig = go.Figure(go.Scatter(y=smoothed_steps, mode='lines', line=dict(color='#ff00ff', width=3)))
        fig.update_layout(title="Steps to Goal", xaxis_title="Episodes", yaxis_title="Steps", **layout_dark)
        st.plotly_chart(fig, use_container_width=True)
    with tab3:
        fig = go.Figure(go.Scatter(y=epsilons, mode='lines', line=dict(color='#00f3ff', width=3)))
        fig.update_layout(title="Epsilon Decay", xaxis_title="Episodes", yaxis_title="Epsilon", **layout_dark)
        st.plotly_chart(fig, use_container_width=True)

def draw_static_discrete_map(env, placeholder):
    html_grid = "<div style='display: grid; grid-template-columns: repeat(10, 45px); gap: 4px; justify-content: center; margin: 20px 0;'>"
    for r in range(env.nrow):
        for c in range(env.ncol):
            cell_content, bg_color = "", "#1f2937"
            if hasattr(env, 'start_state') and (r, c) == env.start_state: cell_content, bg_color = "🐶", "rgba(0, 243, 255, 0.2)"
            elif hasattr(env, 'goal_state') and (r, c) == env.goal_state: cell_content, bg_color = "🍖", "rgba(57, 255, 20, 0.2)"
            elif hasattr(env, 'walls') and (r, c) in env.walls: cell_content, bg_color = "🧱", "#111827"
            elif hasattr(env, 'cliff') and (r, c) in env.cliff: cell_content, bg_color = "☠️", "rgba(255, 0, 0, 0.2)"
            elif hasattr(env, 'traps') and (r, c) in env.traps: cell_content, bg_color = "⚡", "rgba(255, 0, 255, 0.2)"
            elif hasattr(env, 'slippery_cells') and (r, c) in env.slippery_cells: cell_content, bg_color = "💧", "rgba(56, 139, 253, 0.1)"
            
            html_grid += f"<div class='grid-cell' style='background-color: {bg_color}; width: 45px; height: 45px; border: 1px solid #374151;'>{cell_content}</div>"
    html_grid += "</div>"
    placeholder.markdown(html_grid, unsafe_allow_html=True)

def draw_static_continuous_map(env, placeholder):
    fig = go.Figure()
    fig.add_shape(type="circle", x0=env.goal_pos[0]-env.goal_radius, y0=env.goal_pos[1]-env.goal_radius, x1=env.goal_pos[0]+env.goal_radius, y1=env.goal_pos[1]+env.goal_radius, fillcolor="rgba(57, 255, 20, 0.2)", line_color="#39ff14")
    fig.add_trace(go.Scatter(x=[env.goal_pos[0]], y=[env.goal_pos[1]], mode='text', text=['🍖'], textfont=dict(size=26), name='Goal', hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[env.start_pos[0]], y=[env.start_pos[1]], mode='text', text=['🐶'], textfont=dict(size=24), name='Start'))
    
    if hasattr(env, 'obstacles'):
        for obs in env.obstacles:
            fig.add_shape(type="circle", x0=obs[0]-env.obstacle_radius, y0=obs[1]-env.obstacle_radius, x1=obs[0]+env.obstacle_radius, y1=obs[1]+env.obstacle_radius, fillcolor="rgba(255, 0, 0, 0.3)", line_color="#ff0000")
            
    fig.update_layout(xaxis=dict(range=[0, env.room_size], showgrid=True, gridcolor='#1f2937', zeroline=False), yaxis=dict(range=[0, env.room_size], showgrid=True, gridcolor='#1f2937', zeroline=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', width=800, height=550)
    with placeholder.container():
        st.plotly_chart(fig, use_container_width=True)

def visualize_grid_episode(env, policy, max_steps=100):
    state = env.reset() if hasattr(env, 'reset') else 0
    placeholder = st.empty()
    success_placeholder = st.empty()
    
    step = 0
    for step in range(max_steps):
        row, col = state // env.ncol, state % env.ncol
        html_grid = "<div style='display: grid; grid-template-columns: repeat(10, 45px); gap: 4px; justify-content: center; margin: 20px 0;'>"
        
        for r in range(env.nrow):
            for c in range(env.ncol):
                cell_content, bg_color, border, box_shadow = "", "#1f2937", "1px solid #374151", "none"
                if (r, c) == (row, col):
                    cell_content, bg_color, border, box_shadow = "🐶", "rgba(0, 243, 255, 0.2)", "2px solid #00f3ff", "0 0 10px #00f3ff"
                elif hasattr(env, 'goal_state') and (r, c) == env.goal_state:
                    cell_content, bg_color, border, box_shadow = "🍖", "rgba(57, 255, 20, 0.2)", "2px solid #39ff14", "0 0 10px #39ff14"
                elif hasattr(env, 'walls') and (r, c) in env.walls:
                    cell_content, bg_color = "🧱", "#111827"
                elif hasattr(env, 'cliff') and (r, c) in env.cliff:
                    cell_content, bg_color, border = "☠️", "rgba(255, 0, 0, 0.2)", "1px solid #ff0000"
                elif hasattr(env, 'traps') and (r, c) in env.traps:
                    cell_content, bg_color, border = "⚡", "rgba(255, 0, 255, 0.2)", "1px solid #ff00ff"
                elif hasattr(env, 'slippery_cells') and (r, c) in env.slippery_cells:
                    cell_content, bg_color = "💧", "rgba(56, 139, 253, 0.1)"
                else:
                    s = r * env.ncol + c
                    best_action = np.argmax(policy[s])
                    arrow = {0: '↑', 1: '→', 2: '↓', 3: '←'}[best_action]
                    cell_content = f"<span style='font-size: 20px; color: #6b7280; font-weight: bold;'>{arrow}</span>"

                html_grid += f"<div class='grid-cell' style='background-color: {bg_color}; border: {border}; box-shadow: {box_shadow}; width: 45px; height: 45px;'>{cell_content}</div>"
        html_grid += "</div>"
        
        with placeholder.container():
            st.markdown(f"<h3 style='text-align: center; color: #fff;'>צעד: <span style='color:#00f3ff;'>{step}</span></h3>", unsafe_allow_html=True)
            st.markdown(html_grid, unsafe_allow_html=True)
            
        if hasattr(env, 'goal_state') and (row, col) == env.goal_state:
            success_placeholder.markdown(f"<div class='metric-card' style='border-color:#39ff14;'><h4>🎯 משימה הושלמה</h4><h2 style='color:#39ff14;'>הסוכן הגיע ליעד ב-{step} צעדים!</h2></div>", unsafe_allow_html=True)
            break
            
        action = np.argmax(policy[state])
        if hasattr(env, 'step'):
            state, reward, done, _ = env.step(action)
        elif hasattr(env, 'P'):
            prob, next_state, reward, done = env.P[state][action][0]
            state = next_state
        else: break
            
        time.sleep(0.10) 
        if done and reward < 0:
            success_placeholder.markdown(f"<div class='metric-card' style='border-color:#ff0000;'><h4>💥 כישלון אפיזודה</h4><h2 style='color:#ff0000;'>הסוכן התרסק לאחר {step} צעדים.</h2></div>", unsafe_allow_html=True)
            break

def visualize_continuous_path(agent, env_wrapper, room_name):
    state = env_wrapper.reset()
    base_env = env_wrapper.env 
    start_x, start_y = (base_env.agent_pos[0], base_env.agent_pos[1]) if hasattr(base_env, 'agent_pos') else (base_env.state[0], base_env.state[1])
    path_x, path_y = [start_x], [start_y]
    
    step_count = 0
    for _ in range(1200):
        action = agent.select_action(state, epsilon=0.0) 
        state, _, done, _ = env_wrapper.step(action)
        curr_x, curr_y = (base_env.agent_pos[0], base_env.agent_pos[1]) if hasattr(base_env, 'agent_pos') else (base_env.state[0], base_env.state[1])
        path_x.append(curr_x)
        path_y.append(curr_y)
        step_count += 1
        if done: break
            
    fig = go.Figure()
    fig.add_shape(type="circle", x0=base_env.goal_pos[0]-base_env.goal_radius, y0=base_env.goal_pos[1]-base_env.goal_radius, x1=base_env.goal_pos[0]+base_env.goal_radius, y1=base_env.goal_pos[1]+base_env.goal_radius, fillcolor="rgba(57, 255, 20, 0.2)", line_color="#39ff14")
    fig.add_trace(go.Scatter(x=[base_env.goal_pos[0]], y=[base_env.goal_pos[1]], mode='text', text=['🍖'], textfont=dict(size=26), name='Goal', hoverinfo="skip"))
    
    if hasattr(base_env, 'obstacles'):
        for obs in base_env.obstacles:
            fig.add_shape(type="circle", x0=obs[0]-base_env.obstacle_radius, y0=obs[1]-base_env.obstacle_radius, x1=obs[0]+base_env.obstacle_radius, y1=obs[1]+base_env.obstacle_radius, fillcolor="rgba(255, 0, 0, 0.3)", line_color="#ff0000")
            
    fig.add_trace(go.Scatter(x=path_x, y=path_y, mode='lines', name='Path', line=dict(color='#00f3ff', width=4)))
    fig.add_trace(go.Scatter(x=[path_x[0]], y=[path_y[0]], mode='text', text=['🐶'], textfont=dict(size=24), name='Start'))
    fig.add_trace(go.Scatter(x=[path_x[-1]], y=[path_y[-1]], mode='markers', marker=dict(color='#ff00ff', size=12, symbol='x'), name='End'))
    
    fig.update_layout(title=dict(text=f"סימולציית ניווט - סה'כ צעדים: {step_count}", font=dict(color='#00f3ff', size=20)), xaxis=dict(range=[0, base_env.room_size], showgrid=True, gridcolor='#1f2937', zeroline=False), yaxis=dict(range=[0, base_env.room_size], showgrid=True, gridcolor='#1f2937', zeroline=False), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='#e2e8f0'), width=800, height=550, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

def live_grid_callback(ep, path_x, path_y, base_env, placeholder, room_name, episodes, current_reward=0.0, current_eps=0.0):
    curr_x = int(round(max(0, min(base_env.room_size - 1, path_x[-1]))))
    curr_y = int(round(max(0, min(base_env.room_size - 1, path_y[-1]))))
    goal_x = min(int(base_env.goal_pos[0]), int(base_env.room_size - 1))
    goal_y = min(int(base_env.goal_pos[1]), int(base_env.room_size - 1))
    
    grid_ui = "<div style='font-family: monospace; letter-spacing: 5px; text-align: center; font-size: 18px; line-height: 1.5; background: #111827; padding: 20px; border-radius: 10px; border: 1px solid #374151;'>"
    for y in range(int(base_env.room_size)):
        for x in range(int(base_env.room_size)):
            if (x, y) == (curr_x, curr_y): grid_ui += "🐶"
            elif (x, y) == (goal_x, goal_y): grid_ui += "🍖"
            else:
                is_obstacle = False
                if hasattr(base_env, 'obstacles'):
                    for obs in base_env.obstacles:
                        if (x, y) == (int(round(obs[0])), int(round(obs[1]))): is_obstacle = True; break
                grid_ui += "🧱" if is_obstacle else "⬛"
        grid_ui += "<br>"
    grid_ui += "</div>"
    
    metrics_html = render_live_metrics(ep, episodes, current_reward, current_eps)
    
    with placeholder.container():
        st.markdown(metrics_html, unsafe_allow_html=True)
        st.markdown(f"<h3 style='text-align:center;'>תצוגת רדאר חיה: אפיזודה {ep}</h3>", unsafe_allow_html=True)
        st.markdown(grid_ui, unsafe_allow_html=True)

# ==========================================
# כותרת ראשית ותפריט
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo: st.markdown("<h1 style='font-size: 60px;'>🐶</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1>Hezki Escape Room: Reinforcement Learning </h1>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 18px; color: #9ca3af;'>מערכת אימון ובקרה חכמה לסוכן למידת חיזוקים במרחבים שונים</p>", unsafe_allow_html=True)

st.markdown("---")

room_choice = st.sidebar.selectbox(
    "🌐 בחר חדר סימולציה:",
    ["Room 1: Dynamic Programming", "Room 2: SARSA", "Room 3: Q-Learning", "Room 4: Continuous DQN", "Room 5: Double DQN"]
)
st.sidebar.markdown("---")

# ==========================================
# Room 1: Dynamic Programming
# ==========================================
if room_choice == "Room 1: Dynamic Programming":
    st.header("Room 1 - Dynamic Programming (DP)")
    show_room_info("Policy Iteration (Known Model)", "10x10 Grid (100 discrete states)", "4 directions (up, down, right, left)", "Slippery cells creating stochastic uncertainty in movement.")
    st.sidebar.header("⚙️ הייפר-פרמטרים לאימון")
    discount_factor = st.sidebar.slider("Discount Factor (γ)", 0.1, 0.999, 0.99)
    theta = st.sidebar.number_input("Theta (Convergence)", value=0.0001, format="%.5f")
    
    # אתחול ראשוני לסביבה רגילה (אם טרם נוצרה)
    if 'room1_env' not in st.session_state:
        st.session_state.room1_env = room1_DP.Room1Env(randomize_map=False)
        
    # כפתור שמייצר מפה אקראית חדשה בכל לחיצה
    if st.button("🎲 הגרל מפה אקראית (Randomize Map)", key="btn_rand_r1"):
        st.session_state.room1_env = room1_DP.Room1Env(randomize_map=True)
        
    env = st.session_state.room1_env

    map_placeholder = st.empty()
    draw_static_discrete_map(env, map_placeholder)
    
    if st.button("🚀 חשב מדיניות אופטימלית"):
        map_placeholder.empty()
        with st.spinner("מבצע אופטימיזציה..."):
            start_time = time.time()
            optimal_policy, optimal_V, changes_history = room1_DP.policy_iteration(env, discount_factor=discount_factor, theta=theta)
            st.success(f"האלגוריתם התכנס למדיניות אופטימלית תוך {time.time() - start_time:.2f} שניות!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("מפת ערכי המצבים (Value Heatmap)")
                V_matrix = optimal_V.reshape((env.nrow, env.ncol)) 
                fig_heatmap = px.imshow(V_matrix, text_auto=True, color_continuous_scale='Magma', aspect="auto")
                fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
                st.plotly_chart(fig_heatmap, use_container_width=True)
            with col2:
                st.subheader("סימולציה של הסוכן (Optimal Policy)")
                visualize_grid_episode(env, optimal_policy)

# ==========================================
# Room 2: SARSA
# ==========================================
elif room_choice == "Room 2: SARSA":
    st.header("Room 2 - SARSA (On-Policy)")
    show_room_info("SARSA", "10x10 Grid (100 discrete states)", "4 directions", "Unknown environment model, heavy traps (penalty -50) combined with slippery cells.")
    st.sidebar.header("⚙️ הייפר-פרמטרים לאימון")
    episodes = st.sidebar.slider("Episodes", 100, 3000, 1000, step=100)
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.8, 0.999, 0.99)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.995)
    save_freq = st.sidebar.slider("תדירות שמירת מצבים (Checkpoints)", 50, 500, 100, step=50)
    
    if 'room2_env' not in st.session_state:
        st.session_state.room2_env = room2_Sarsa.Room2Env(randomize_map=False)
        
    if st.button("🎲 הגרל מפה אקראית (Randomize Map)", key="btn_rand_r2"):
        st.session_state.room2_env = room2_Sarsa.Room2Env(randomize_map=True)
        
    env = st.session_state.room2_env

    map_placeholder = st.empty()
    draw_static_discrete_map(env, map_placeholder)
        
    if st.button("🚀 התחל אימון SARSA"):
        map_placeholder.empty()
        with st.spinner("הסוכן חוקר את הסביבה..."):
            def live_cb(ep, Q_table, current_env):
                learned_policy = np.zeros([current_env.nS, current_env.nA])
                for s in range(current_env.nS): learned_policy[s, np.argmax(Q_table[s])] = 1.0
                map_placeholder.empty()
                with map_placeholder.container():
                    st.markdown(f"<h4 style='text-align:center;'>צופה בלמידה - אפיזודה {ep}</h4>", unsafe_allow_html=True)
                    visualize_grid_episode(current_env, learned_policy, max_steps=40)

            Q, rewards, steps, epsilons, checkpoints = room2_Sarsa.sarsa(env, num_episodes=episodes, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, save_freq=save_freq, live_callback=live_cb)
            st.session_state.room2_data = (env, Q, rewards, steps, epsilons, checkpoints)
            
    if 'room2_data' in st.session_state and st.session_state.room2_data is not None:
        env, Q, rewards, steps, epsilons, checkpoints = st.session_state.room2_data
        st.success("אימון SARSA הושלם בהצלחה!")
        st.subheader("גרפי התכנסות")
        plot_training_metrics(rewards, steps, epsilons)

        st.markdown("---")
        st.subheader("🔥 מפת ערכי המצבים (Value Heatmap)")
        
        # חישוב ערך המקסימום לכל מצב מתוך טבלת ה-Q
        V_matrix = np.max(Q, axis=1).reshape((env.nrow, env.ncol))
        
        fig_heatmap = px.imshow(V_matrix, text_auto=".1f", color_continuous_scale='Magma', aspect="auto")
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")
        st.subheader("⏳ מכונת הזמן: שחזור אפיזודות (Episode Replay)")
        checkpoints_list = sorted(list(checkpoints.keys()))
        selected_ep = st.select_slider("אפיזודה לבדיקה:", options=checkpoints_list, value=checkpoints_list[-1])
        if st.button("▶️ צפה באנימציה"):
            Q_selected = checkpoints[selected_ep]
            learned_policy = np.zeros([env.nS, env.nA])
            for s in range(env.nS): learned_policy[s, np.argmax(Q_selected[s])] = 1.0
            visualize_grid_episode(env, learned_policy)

# ==========================================
# Room 3: Q-Learning
# ==========================================
elif room_choice == "Room 3: Q-Learning":
    st.header("Room 3 - Q-Learning (Off-Policy)")
    show_room_info("Q-Learning", "10x10 Grid (100 discrete states)", "4 directions", "Dangerous cliff (penalty -100) creating a dilemma between a shortcut and a safe detour.")
    st.sidebar.header("⚙️ הייפר-פרמטרים לאימון")
    episodes = st.sidebar.slider("Episodes", 100, 3000, 1000, step=100)
    alpha = st.sidebar.slider("Learning Rate (α)", 0.01, 1.0, 0.1)
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.8, 0.999, 0.99)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.9, 0.999, 0.995)
    save_freq = st.sidebar.slider("תדירות שמירת מצבים (Checkpoints)", 50, 500, 100, step=50)
    
    if 'room3_env' not in st.session_state:
        st.session_state.room3_env = room3_Qlearning.Room3Env(randomize_map=False)
        
    if st.button("🎲 הגרל מפה אקראית (Randomize Map)", key="btn_rand_r3"):
        st.session_state.room3_env = room3_Qlearning.Room3Env(randomize_map=True)
        
    env = st.session_state.room3_env

    map_placeholder = st.empty()
    draw_static_discrete_map(env, map_placeholder)
        
    if st.button("🚀 התחל אימון Q-Learning"):
        map_placeholder.empty()
        with st.spinner("הסוכן לומד לקפוץ מעל צוקים..."):
            def live_cb(ep, Q_table, current_env):
                learned_policy = np.zeros([current_env.nS, current_env.nA])
                for s in range(current_env.nS): learned_policy[s, np.argmax(Q_table[s])] = 1.0
                map_placeholder.empty()
                with map_placeholder.container():
                    st.markdown(f"<h4 style='text-align:center;'>צופה בלמידה - אפיזודה {ep}</h4>", unsafe_allow_html=True)
                    visualize_grid_episode(current_env, learned_policy, max_steps=40)

            Q, rewards, steps, epsilons, checkpoints = room3_Qlearning.q_learning(env, num_episodes=episodes, alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay, save_freq=save_freq, live_callback=live_cb)
            st.session_state.room3_data = (env, Q, rewards, steps, epsilons, checkpoints)
            
    if 'room3_data' in st.session_state and st.session_state.room3_data is not None:
        env, Q, rewards, steps, epsilons, checkpoints = st.session_state.room3_data
        st.success("אימון Q-Learning הושלם בהצלחה!")
        st.subheader("גרפי התכנסות")
        plot_training_metrics(rewards, steps, epsilons)

        st.markdown("---")
        st.subheader("🔥 מפת ערכי המצבים (Value Heatmap)")
        
        # חישוב ערך המקסימום לכל מצב מתוך טבלת ה-Q
        V_matrix = np.max(Q, axis=1).reshape((env.nrow, env.ncol))
        
        fig_heatmap = px.imshow(V_matrix, text_auto=".1f", color_continuous_scale='Magma', aspect="auto")
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")
        st.subheader("⏳ מכונת הזמן: שחזור אפיזודות (Episode Replay)")
        checkpoints_list = sorted(list(checkpoints.keys()))
        selected_ep = st.select_slider("אפיזודה לבדיקה:", options=checkpoints_list, value=checkpoints_list[-1])
        if st.button("▶️ צפה באנימציה"):
            Q_selected = checkpoints[selected_ep]
            learned_policy = np.zeros([env.nS, env.nA])
            for s in range(env.nS): learned_policy[s, np.argmax(Q_selected[s])] = 1.0
            visualize_grid_episode(env, learned_policy)

# ==========================================
# Room 4: Continuous DQN
# ==========================================
elif room_choice == "Room 4: Continuous DQN":
    st.header("Room 4 - Deep Q-Network in Continuous Space")
    show_room_info("Deep Q-Network (DQN)", "Continuous: 10x10 meters, Position (X,Y) and Velocity (Vx,Vy)", "9 velocity combinations (discretized actions)", "Value function approximation in infinite space, neural networks, and Reward Shaping.")
    st.sidebar.header("⚙️ הייפר-פרמטרים לאימון")
    episodes = st.sidebar.slider("Episodes", 200, 2000, 800, step=50)
    learning_rate = st.sidebar.number_input("Learning Rate (α)", min_value=0.0001, max_value=0.01, value=0.001, step=0.0001, format="%.4f")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.80, 0.99, 0.99)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.900, 0.999, 0.992, step=0.001)
    target_update_freq = st.sidebar.slider("Target Net Update Freq", 50, 500, 100, step=50)
    save_freq = st.sidebar.slider("תדירות שמירת מצבים (Checkpoints)", 50, 200, 50, step=50)
    
    if 'room4_base_env' not in st.session_state:
        st.session_state.room4_base_env = room4_DQN.Room4Env(randomize_map=False)
        
    if st.button("🎲 הגרל מפה אקראית (Randomize Map)", key="btn_rand_r4"):
        st.session_state.room4_base_env = room4_DQN.Room4Env(randomize_map=True)
        
    base_env = st.session_state.room4_base_env

    map_placeholder = st.empty()
    draw_static_continuous_map(base_env, map_placeholder)
        
    if st.button("🧠 התחל אימון רשת נוירונים (DQN)"):
        map_placeholder.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_plot_placeholder = st.empty()
        
        with st.spinner("מאמן רשת נוירונים עמוקה..."):
            optimized_env = room4_DQN.Room4OptimizedWrapper(base_env)
            def callback(ep, px_arr, py_arr, env_obj, current_r, current_e):
                live_grid_callback(ep, px_arr, py_arr, env_obj, live_plot_placeholder, "Continuous DQN", episodes, current_r, current_e)

            agent, rewards, steps, epsilons, checkpoints = room4_DQN.train_dqn(
                optimized_env, episodes=episodes, learning_rate=learning_rate, gamma=gamma,
                batch_size=batch_size, epsilon_decay=epsilon_decay, save_freq=save_freq, 
                progress_bar=progress_bar, status_text=status_text, live_callback=callback, target_update_freq=target_update_freq
            )
            st.session_state.room4_data = (agent, optimized_env, rewards, steps, epsilons, checkpoints)
            
            st.success("אימון מודל AI הושלם!")
            live_plot_placeholder.empty()
            
    if 'room4_data' in st.session_state and st.session_state.room4_data is not None:
        agent, optimized_env, rewards, steps, epsilons, checkpoints = st.session_state.room4_data
        st.subheader("ניתוח ביצועים")
        plot_training_metrics(rewards, steps, epsilons, window=20)

        st.markdown("---")
        st.subheader("🔥 מפת ערכי המצבים (Heatmap - חזוי רשת נוירונים)")
        
        grid_resolution = 20 # יצירת גריד של 20x20
        x_vals = np.linspace(0, optimized_env.max_x, grid_resolution)
        y_vals = np.linspace(0, optimized_env.max_y, grid_resolution)
        V_matrix = np.zeros((grid_resolution, grid_resolution))
        
        agent.q_network.eval()
        with torch.no_grad():
            states_list = []
            for y in y_vals:
                for x in x_vals:
                    dummy_state = np.array([x, y, 0.0, 0.0])
                    states_list.append(optimized_env._normalize_state(dummy_state))
            
            states_tensor = torch.tensor(np.array(states_list), dtype=torch.float32)
            q_values = agent.q_network(states_tensor)
            max_q_values = torch.max(q_values, dim=1)[0].numpy()
            
            V_matrix = max_q_values.reshape((grid_resolution, grid_resolution))
                    
        fig_heatmap = px.imshow(V_matrix, x=x_vals, y=y_vals, color_continuous_scale='Magma', origin='lower')

        goal_pos = optimized_env.env.goal_pos
        fig_heatmap.add_trace(go.Scatter(x=[goal_pos[0]], y=[goal_pos[1]],mode='text', text=['🍖'],textfont=dict(size=26),name='Goal'))
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")
        st.subheader("⏳ מכונת הזמן: שחזור משקלי רשת (Model Replay)")
        checkpoints_list = sorted(list(checkpoints.keys()))
        selected_ep = st.select_slider("אפיזודה לבדיקה:", options=checkpoints_list, value=checkpoints_list[-1])
        if st.button("▶️ הרץ סימולציית מסלול"):
            agent.q_network.load_state_dict(checkpoints[selected_ep])
            visualize_continuous_path(agent, optimized_env, f"Continuous DQN (אפיזודה {selected_ep})")

# ==========================================
# Room 5: Double DQN
# ==========================================
elif room_choice == "Room 5: Double DQN":
    st.header("Room 5 - Double DQN with Vision and Dynamic Obstacles")
    show_room_info("Double DQN", "20-dimensional vector (X, Y, Vx, Vy + 16 Raycast sensors)", "9 velocity combinations", "Dodge dynamically generated obstacles and prevent overestimation using DDQN.")
    st.sidebar.header("⚙️ הייפר-פרמטרים לאימון")
    episodes = st.sidebar.slider("Episodes", 200, 3000, 1500, step=100)
    learning_rate = st.sidebar.number_input("Learning Rate (α)", min_value=0.0001, max_value=0.01, value=0.0005, step=0.0001, format="%.4f")
    gamma = st.sidebar.slider("Discount Factor (γ)", 0.80, 0.99, 0.99)
    batch_size = st.sidebar.selectbox("Batch Size", [32, 64, 128], index=1)
    epsilon_decay = st.sidebar.slider("Epsilon Decay", 0.900, 0.999, 0.997, step=0.001)
    target_update_freq = st.sidebar.slider("Target Net Update Freq", 50, 500, 200, step=50)
    save_freq = st.sidebar.slider("תדירות שמירת מצבים (Checkpoints)", 50, 200, 50, step=50)
    
    if 'room5_base_env' not in st.session_state:
        st.session_state.room5_base_env = room5_DoubleDQN.Room5Env(num_obstacles=(5, 10), randomize_map=False)
        
    if st.button("🎲 הגרל מפה אקראית (Randomize Map)", key="btn_rand_r5"):
        st.session_state.room5_base_env = room5_DoubleDQN.Room5Env(num_obstacles=(5, 10), randomize_map=True)
        
    base_env = st.session_state.room5_base_env

    optimized_env = room5_DoubleDQN.Room5OptimizedWrapper(base_env)

    map_placeholder = st.empty()
    draw_static_continuous_map(base_env, map_placeholder)
        
    if st.button("☢️ הפעל אימון רשת כפולה (DDQN)"):
        map_placeholder.empty()
        progress_bar = st.progress(0)
        status_text = st.empty()
        live_plot_placeholder = st.empty()
        
        with st.spinner("מריץ סביבה עוינת עם מכשולים דינמיים..."):
            def callback(ep, px_arr, py_arr, env_obj, current_r, current_e):
                live_grid_callback(ep, px_arr, py_arr, env_obj, live_plot_placeholder, "Double DQN", episodes, current_r, current_e)

            agent, rewards, steps, epsilons, checkpoints = room5_DoubleDQN.train_room5(
                optimized_env, episodes=episodes, learning_rate=learning_rate, gamma=gamma,
                batch_size=batch_size, epsilon_decay=epsilon_decay, save_freq=save_freq, 
                progress_bar=progress_bar, status_text=status_text, live_callback=callback, target_update_freq=target_update_freq
            )
            st.session_state.room5_data = (agent, rewards, steps, epsilons, checkpoints)
            
            st.success("אימון מודל AI הושלם!")
            live_plot_placeholder.empty()

    if 'room5_data' in st.session_state and st.session_state.room5_data is not None:
        agent, rewards, steps, epsilons, checkpoints = st.session_state.room5_data
        st.subheader("ניתוח ביצועים")
        plot_training_metrics(rewards, steps, epsilons, window=50)

        st.markdown("---")
        st.subheader("🔥 מפת ערכי המצבים בהתחשב במכשולים הנוכחיים")
        
        grid_resolution = 25
        x_vals = np.linspace(0, base_env.room_size, grid_resolution)
        y_vals = np.linspace(0, base_env.room_size, grid_resolution)
        V_matrix = np.zeros((grid_resolution, grid_resolution))
        
        agent.q_network.eval()
        with torch.no_grad():
            obs_list = []
            for y in y_vals:
                for x in x_vals:
                    base_env.agent_pos = np.array([x, y], dtype=np.float32)
                    base_env.agent_vel = np.array([0.0, 0.0], dtype=np.float32)
                    raw_obs = base_env._get_obs()
                    obs_list.append(optimized_env._normalize_state(raw_obs))
            
            states_tensor = torch.tensor(np.array(obs_list), dtype=torch.float32)
            q_values = agent.q_network(states_tensor)
            max_q_values = torch.max(q_values, dim=1)[0].numpy()
            
            V_matrix = max_q_values.reshape((grid_resolution, grid_resolution))
                    
        
        fig_heatmap = go.Figure(data=go.Heatmap(z=V_matrix, x=x_vals, y=y_vals, colorscale='Magma'))
        
        # ציור המכשולים האקראיים על גבי מפת החום כדי לראות איך הרשת מתייחסת אליהם
        for obs in base_env.obstacles:
            fig_heatmap.add_shape(type="circle", 
                                  x0=obs[0]-base_env.obstacle_radius, y0=obs[1]-base_env.obstacle_radius, 
                                  x1=obs[0]+base_env.obstacle_radius, y1=obs[1]+base_env.obstacle_radius, 
                                  line_color="cyan", fillcolor="rgba(0, 255, 255, 0.2)")

        goal_pos = base_env.goal_pos
        fig_heatmap.add_trace(go.Scatter( x=[goal_pos[0]], y=[goal_pos[1]],mode='text', text=['🍖'],textfont=dict(size=26),name='Goal'))
        fig_heatmap.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'),xaxis_title='X Position', yaxis_title='Y Position')
        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.markdown("---")
        st.subheader("⏳ מכונת הזמן: שחזור משקלי רשת בחדר רנדומלי (Model Replay)")
        checkpoints_list = sorted(list(checkpoints.keys()))
        selected_ep = st.select_slider("אפיזודה לבדיקה:", options=checkpoints_list, value=checkpoints_list[-1])
        if st.button("▶️ הרץ סימולציה בחדר אקראי חדש"):
            agent.q_network.load_state_dict(checkpoints[selected_ep])
            
            new_base_env = room5_DoubleDQN.Room5Env(num_obstacles=(5, 10))
            new_optimized_env = room5_DoubleDQN.Room5OptimizedWrapper(new_base_env)
            visualize_continuous_path(agent, new_optimized_env, f"Double DQN (אפיזודה {selected_ep})")

# python3 -m streamlit run app.py
