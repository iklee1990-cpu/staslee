# train_dqn_intercept.py
import os
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback
import matplotlib.pyplot as plt

from drone_intercept_env import DroneIntercept1D

# ---------------------------------------
# 1) 환경 래핑(모니터)
# ---------------------------------------
def make_env(seed=42):
    env = DroneIntercept1D(L=20, max_steps=100, target_speed=1, render_mode="ansi", seed=seed)
    env = Monitor(env)  # 에피소드 리워드/길이 기록
    return env

# ---------------------------------------
# 2) 학습
# ---------------------------------------
def train_model(total_timesteps=50_000, model_path="dqn_intercept.zip"):
    env = make_env(seed=42)
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        verbose=1,
        tensorboard_log=None,  # 필요하면 경로 지정
        exploration_fraction=0.3,     # ε decay 비율
        exploration_initial_eps=1.0,  # 초기 ε
        exploration_final_eps=0.02    # 최종 ε
    )

    # 평가 콜백: 1000 스텝마다 10에피소드 평가, 최고 모델 저장
    eval_env = make_env(seed=123)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./best_model",
        log_path="./logs",
        eval_freq=1_000,
        n_eval_episodes=10,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=total_timesteps, callback=eval_callback)
    model.save(model_path)
    env.close()
    eval_env.close()
    return model_path

# ---------------------------------------
# 3) 평가 및 성공률 계산
# ---------------------------------------
def evaluate(model_path="dqn_intercept.zip", n_eval_episodes=100):
    env = make_env(seed=7)
    model = DQN.load(model_path, env=env)

    # 평균 리워드/표준편차
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=True)
    print(f"[평균 리워드] {mean_reward:.2f} ± {std_reward:.2f} (N={n_eval_episodes})")

    # 성공률 계산: terminated(True)가 capture 상황
    success = 0
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated:
                success += 1
                break
            if truncated:
                break
    env.close()
    print(f"[성공률] {success}/{n_eval_episodes} = {success/n_eval_episodes*100:.1f}%")

# ---------------------------------------
# 4) 데모(한 에피소드) & 텍스트 렌더 보기
# ---------------------------------------
def demo_episode(model_path="dqn_intercept.zip"):
    env = make_env(seed=None)
    model = DQN.load(model_path, env=env)
    obs, _ = env.reset()

    frames = []
    while True:
        frame = env.render()
        frames.append(frame)
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break
    env.close()

    print("\n=== DEMO (ansi 텍스트) ===")
    for f in frames[:10]:  # 앞부분 10프레임만 미리보기
        print(f)
    # 전체 프레임을 파일로 저장
    with open("demo_ansi.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(frames))
    print("\n[저장] demo_ansi.txt (전체 프레임)")

# ---------------------------------------
# 5) 학습 곡선(에피소드 리워드) 간단 시각화
#    Monitor는 ./monitor.csv 형태로 저장하므로, 간단히 이동평균을 그립니다.
# ---------------------------------------
def plot_rewards(log_csv_path="./monitor.csv", window=20, out_path="reward_curve.png"):
    if not os.path.exists(log_csv_path):
        # Monitor가 기본 파일명을 랜덤 접두사로 쓰는 경우가 있어, 탐색해서 첫 파일 사용
        mons = [f for f in os.listdir(".") if f.endswith(".monitor.csv") or f.endswith("monitor.csv")]
        if mons:
            log_csv_path = mons[0]
        else:
            print("[경고] monitor 로그 파일을 찾지 못했습니다.")
            return
    # Monitor CSV: 첫 줄(메타데이터) 건너뛰고 로드
    rewards = []
    with open(log_csv_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):  # 주석/메타
                continue
            parts = line.strip().split(",")
            # columns: r,l,t
            if len(parts) >= 3:
                try:
                    r = float(parts[0])
                    rewards.append(r)
                except:
                    pass
    if not rewards:
        print("[경고] 보상 기록이 비었습니다.")
        return

    # 이동평균
    rewards = np.array(rewards)
    if len(rewards) >= window:
        ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
    else:
        ma = rewards

    plt.figure()
    plt.plot(rewards, label="Episode reward")
    plt.plot(range(window-1, window-1+len(ma)), ma, label=f"Moving Avg({window})")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[저장] {out_path}")

# ---------------------------------------
# main
# ---------------------------------------
if __name__ == "__main__":
    model_path = train_model(total_timesteps=50_000, model_path="dqn_intercept.zip")
    evaluate(model_path, n_eval_episodes=100)
    demo_episode(model_path)
    plot_rewards()
