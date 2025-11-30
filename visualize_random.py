# visualize_random.py
"""
실험1: 학습 없이 랜덤 행동만 하는 에이전트 → 실패 GIF 생성
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from drone_intercept_env import DroneIntercept1D

# 환경 불러오기
env = DroneIntercept1D(render_mode="ansi", seed=42)

# 위치 기록
states_i = []
states_t = []

obs, _ = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action = env.action_space.sample()  # 랜덤 행동
    obs, reward, terminated, truncated, info = env.step(action)

    states_i.append(env.interceptor_pos)
    states_t.append(env.target_pos)

env.close()

# 시각화
fig, ax = plt.subplots(figsize=(8,2))
L = env.L

def update(frame):
    ax.clear()
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.scatter(states_i[frame], 0, s=200, label="Interceptor (I)")
    ax.scatter(states_t[frame], 0, s=200, label="Target (T)", marker="x")
    ax.set_title(f"Random Policy – Frame {frame+1}/{len(states_i)}")
    ax.legend(loc="upper right")

ani = FuncAnimation(fig, update, frames=len(states_i), interval=200)
ani.save("intercept_random.gif", dpi=100, writer="pillow")
print("[저장 완료] intercept_random.gif")
