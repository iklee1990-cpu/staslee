# 1D Drone Interception using Reinforcement Learning

강화학습(DQN)을 이용한 1차원 드론 요격 텀프로젝트입니다.

## 주요 파일
- `drone_intercept_env.py` : 1D 드론 요격 Gym 환경 정의
- `train_dqn_intercept.py` : 정지 드론(target_speed=0) DQN 학습
- `train_dqn_intercept_moving.py` : 이동 드론(target_speed=1) DQN 학습
- `visualize_random.py` : 랜덤 정책 GIF 생성
- `visualize_intercept.py`, `visualize_moving.py` : 학습 결과 GIF 생성

## 간단 실행 방법
```bash
python train_dqn_intercept.py
python train_dqn_intercept_moving.py
python visualize_moving.py
