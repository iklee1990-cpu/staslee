# train_dqn_intercept_moving.py
"""
실험 2: 이동하는 드론(target_speed=1) 요격 실험
 - 기존 실험: target_speed=0 (정지 드론) -> 성공률 100%
 - 이번 실험: target_speed=1 (오른쪽으로 한 칸씩 이동)
를 대상으로 DQN 학습 성능을 비교하기 위한 코드
"""

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from drone_intercept_env import DroneIntercept1D


# ---------------------------------------
# 1) 환경 생성 함수 (이동 드론 버전)
# ---------------------------------------
def make_env(seed=42):
    """
    target_speed=1인 이동 드론 환경 생성
    L, max_steps는 기존과 동일하게 사용
    """
    env = DroneIntercept1D(
        L=10,
        max_steps=50,
        target_speed=1,       # ★ 여기만 다름: 드론이 매 스텝 1칸씩 이동
        render_mode="ansi",
        seed=seed
    )
    env = Monitor(env)        # 에피소드 리워드 기록용
    return env


# ---------------------------------------
# 2) 학습 함수
# ---------------------------------------
def train_model(total_timesteps=200_000, model_path="dqn_intercept_moving.zip"):
    env = make_env(seed=42)

    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=5e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=1_000,
        verbose=1,
        exploration_fraction=0.5,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()
    return model_path


# ---------------------------------------
# 3) 평가 함수 (평균 리워드 + 성공률)
# ---------------------------------------
def evaluate_model(model_path="dqn_intercept_moving.zip", n_eval_episodes=100):
    env = make_env(seed=123)
    model = DQN.load(model_path, env=env)

    # 평균 리워드 / 표준편차
    mean_reward, std_reward = evaluate_policy(
        model, env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )
    print(f"[이동 드론] 평균 리워드: {mean_reward:.2f} ± {std_reward:.2f} (N={n_eval_episodes})")

    # 성공률 계산
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
    rate = success / n_eval_episodes * 100.0
    print(f"[이동 드론] 성공률: {success}/{n_eval_episodes} = {rate:.1f}%")


# ---------------------------------------
# 4) 데모(텍스트) 몇 프레임 출력
# ---------------------------------------
def demo_episode(model_path="dqn_intercept_moving.zip"):
    env = make_env(seed=7)
    model = DQN.load(model_path, env=env)
    obs, _ = env.reset()

    frames = []
    while True:
        frames.append(env.render())
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            frames.append(env.render())
            break

    env.close()

    print("\n=== 이동 드론 DEMO (ansi 텍스트 일부) ===")
    for f in frames[:10]:
        print(f)

    with open("demo_ansi_moving.txt", "w", encoding="utf-8") as fp:
        fp.write("\n".join(frames))
    print("[저장] demo_ansi_moving.txt")


# ---------------------------------------
# main
# ---------------------------------------
if __name__ == "__main__":
    # 1) 학습
    model_path = train_model(
        total_timesteps=200_000,
        model_path="dqn_intercept_moving.zip"
    )

    # 2) 평가
    evaluate_model(model_path, n_eval_episodes=100)

    # 3) 데모 출력
    demo_episode(model_path)
