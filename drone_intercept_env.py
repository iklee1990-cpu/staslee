# drone_intercept_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DroneIntercept1D(gym.Env):
    """
    1D 드론 요격 환경
    - 선형 트랙: 0..L
    - 관측: [interceptor_pos_norm, target_pos_norm] in [0,1]^2 (Box)
    - 행동: 0=왼(-1), 1=정지(0), 2=오른(+1) (Discrete(3))
    - 보상: capture +10 / escape or time-out -10 / step -0.05 / -0.01*distance
    - 종료: capture -> terminated=True, escape/time-limit -> truncated=True
    """
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, L: int = 10, max_steps: int = 50, target_speed: int = 0, render_mode: str | None = None, seed: int | None = None):
        super().__init__()
        self.L = L
        self.max_steps = max_steps
        self.target_speed = target_speed
        self.render_mode = render_mode

        # 관측: [interceptor_pos_norm, target_pos_norm]
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(2,), dtype=np.float32)
        # 행동: 0(왼), 1(정지), 2(오른)
        self.action_space = spaces.Discrete(3)

        self.np_random = np.random.default_rng(seed)
        self._reset_internal()

    def _reset_internal(self):
        # 위치 초기화: 서로 다른 위치에서 시작
        self.interceptor_pos = int(self.np_random.integers(low=0, high=self.L // 2))
        self.target_pos = int(self.np_random.integers(low=self.L // 2, high=self.L))
        self.steps = 0

    def _get_obs(self):
        return np.array([self.interceptor_pos / self.L, self.target_pos / self.L], dtype=np.float32)

    def reset(self, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        self._reset_internal()
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: int):
        # 행동 적용: interceptor 이동
        delta = (-1 if action == 0 else (0 if action == 1 else 1))
        self.interceptor_pos = int(np.clip(self.interceptor_pos + delta, 0, self.L))

        # target 이동 (오른쪽으로 고정 속도)
        self.target_pos = int(np.clip(self.target_pos + self.target_speed, 0, self.L))

        self.steps += 1

        # 보상 계산
        distance = abs(self.interceptor_pos - self.target_pos)
        reward = - 0.01

        terminated = False
        truncated = False

        # 요격 성공
        if self.interceptor_pos == self.target_pos:
            reward += 5.0
            terminated = True

        # 드론 탈출(오른쪽 끝) 또는 시간초과
        if self.steps >= self.max_steps:
            truncated = True

        obs = self._get_obs()
        info = {"distance": distance, "step": self.steps}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        # 텍스트(ansi) 렌더링: I=interceptor, T=target
        line = ["_"] * (self.L + 1)
        i = int(self.interceptor_pos); t = int(self.target_pos)
        if 0 <= i <= self.L: line[i] = "I"
        if 0 <= t <= self.L:
            line[t] = "T" if t != i else "X"  # 같은 위치면 X (요격)
        return "".join(line)

    def close(self):
        pass
