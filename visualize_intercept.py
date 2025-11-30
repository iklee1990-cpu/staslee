# visualize_intercept.py
"""
학습된 DQN 모델을 활용하여 드론 요격 과정을
프레임 단위로 기록하고 GIF 애니메이션으로 저장하는 파일.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from stable_baselines3 import DQN
from drone_intercept_env import DroneIntercept1D

# -----------------------------
# 1) 환경 & 학습된 모델 불러오기
# -----------------------------
env = DroneIntercept1D(render_mode="ansi", seed=42)
model = DQN.load("dqn_intercept.zip", env=env)  # 학습된 모델 파일

# -----------------------------
# 2) 평가(1 에피소드) - 위치 기록
# -----------------------------
states_i = []
states_t = []

obs, _ = env.reset()
terminated = False
truncated = False

while not (terminated or truncated):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    # 위치 저장
    states_i.append(env.interceptor_pos)
    states_t.append(env.target_pos)

env.close()

print(f"총 프레임 수: {len(states_i)}")

# -----------------------------
# 3) 애니메이션 시각화
# -----------------------------
fig, ax = plt.subplots(figsize=(8, 2))
L = env.L

def update(frame):
    ax.clear()
    ax.set_xlim(0, L)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])  #  y축 제거 (1차원 표현)

    i_pos = states_i[frame]
    t_pos = states_t[frame]

    # 요격기 (파랑), 드론(빨강)
    ax.scatter(i_pos, 0, s=200, label="Interceptor (I)")
    ax.scatter(t_pos, 0, s=200, label="Target (T)", marker='x')

    ax.set_title(f"Frame {frame+1}/{len(states_i)}")
    ax.legend(loc="upper right")

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=len(states_i), interval=200)

# -----------------------------
# 4) GIF 저장
# -----------------------------
ani.save("intercept_demo.gif", dpi=100, writer="pillow")
print("[완료] intercept_demo.gif 파일 생성됨!")
