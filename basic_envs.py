import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TwoAZeroObsOneStep(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(2)
        self.accion_buena = 0
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finished = False
        obs = 0
        return obs, {}

    def step(self, action):
        reward = 1 if action == self.accion_buena else -1
        self.finished = True
        obs = 0
        return obs, reward, self.finished, False, {}


class TwoARandomObsOneStep(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(2)
        self.finished = False
        self.accion_buena = {0: 0, 1: 1}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finished = False
        self.state = self.np_random.integers(0, 2)
        return self.state, {}

    def step(self, action):
        accion_buena = self.accion_buena[self.state]
        reward = 1 if action == accion_buena else -1
        self.finished = True
        return self.state, reward, self.finished, False, {}


class LineWorldEasyEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(6)
        self.action_space = spaces.Discrete(2)
        self.initial_state = 0
        self.goal_state = 5
        self.state = self.initial_state
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state
        self.finished = False
        return self.state, {}

    def step(self, action):
        if self.finished:
            raise RuntimeError("Episode already ended.")
        # movimiento
        if action == 0 and self.state > 0:
            self.state -= 1
        elif action == 1 and self.state < self.goal_state:
            self.state += 1
        # recompensa
        if self.state == self.goal_state:
            reward = 1
            self.finished = True
        else:
            reward = 0
        return self.state, reward, self.finished, False, {}


class LineWorldMirrorEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(4)
        self.action_space = spaces.Discrete(2)
        self.initial_state = 0
        self.goal_state = 3
        self.state = self.initial_state
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.initial_state
        self.finished = False
        return self.state, {}

    def step(self, action):
        if self.finished:
            raise RuntimeError("")
        if self.state == 1:
            if action == 0:
                self.state += 1
            elif action == 1:
                self.state = max(0, self.state - 1)
        else:
            if action == 0 and self.state > 0:
                self.state -= 1
            elif action == 1 and self.state < self.goal_state:
                self.state += 1
        reward = -1
        if self.state == self.goal_state:
            self.finished = True
        return self.state, reward, self.finished, False, {}



########################################################


class ConstantRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(1)
        self.action_space = spaces.Discrete(1)
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finished = False
        obs = 0
        return obs, {}

    def step(self, action):
        reward = 1.0
        self.finished = True
        obs = 0
        return obs, reward, self.finished, False, {}


class RandomObsBinaryRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.finished = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.finished = False
        self.state = int(self.np_random.choice([0, 1]))
        obs = self._decode_obs(self.state)
        return obs, {}

    def _decode_obs(self, state):
        return -1 if state == 0 else 1

    def step(self, action):
        obs = self._decode_obs(self.state)
        reward = float(obs)
        self.finished = True
        return obs, reward, self.finished, False, {}


class TwoStepDelayedRewardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Discrete(2)
        self.action_space = spaces.Discrete(1)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        obs = 0
        return obs, {}

    def step(self, action):
        self.step_count += 1
        if self.step_count == 1:
            obs = 0
            reward = 0.0
            finished = False
        else:
            obs = 1
            reward = 1.0
            finished = True
        return obs, reward, finished, False, {}
