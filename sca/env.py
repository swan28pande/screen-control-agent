"""Synthetic GUI environment: click the button whose color matches the task bar.

Observation: (3, H, W) uint8 screenshot. Row band at the top is the "instruction"
(target color); buttons are colored rectangles; the cursor is a white cross.
Action: MultiDiscrete([9, 2]) = (move direction: stay + 8 compass dirs, click: no/yes).
"""
import gymnasium as gym
import numpy as np

PALETTE = np.array([[230, 60, 60], [60, 200, 80], [70, 110, 240], [240, 200, 40],
                    [200, 80, 220], [40, 210, 210]], dtype=np.uint8)
DIRS = np.array([[0, 0], [0, -1], [1, -1], [1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1]])  # (dx, dy)
BAR_H, BTN_W, BTN_H = 4, 10, 7


class ScreenClickEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, size=64, n_buttons=4, max_steps=40, step_px=3, shaping=0.0):
        self.size, self.n_buttons, self.max_steps, self.step_px, self.shaping = size, n_buttons, max_steps, step_px, shaping
        self.observation_space = gym.spaces.Box(0, 255, (3, size, size), np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([len(DIRS), 2])

    def _place_buttons(self):
        boxes = []
        while len(boxes) < self.n_buttons:
            x = int(self.np_random.integers(1, self.size - BTN_W - 1))
            y = int(self.np_random.integers(BAR_H + 2, self.size - BTN_H - 1))
            if all(abs(x - bx) > BTN_W + 1 or abs(y - by) > BTN_H + 1 for bx, by in boxes):
                boxes.append((x, y))
        return np.array(boxes)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.buttons = self._place_buttons()
        self.colors = self.np_random.choice(len(PALETTE), self.n_buttons, replace=False)
        self.target = int(self.np_random.integers(self.n_buttons))
        self.cursor = self.np_random.integers([0, BAR_H], [self.size, self.size]).astype(np.int64)
        self.steps = 0
        return self._obs(), {}

    def target_center(self):
        x, y = self.buttons[self.target]
        return np.array([x + BTN_W // 2, y + BTN_H // 2])

    def _hit(self):
        cx, cy = self.cursor
        for i, (x, y) in enumerate(self.buttons):
            if x <= cx < x + BTN_W and y <= cy < y + BTN_H:
                return i
        return None

    def _obs(self):
        img = np.full((self.size, self.size, 3), 24, np.uint8)
        img[:BAR_H] = PALETTE[self.colors[self.target]]
        for (x, y), c in zip(self.buttons, self.colors):
            img[y:y + BTN_H, x:x + BTN_W] = PALETTE[c]
        cx, cy = self.cursor
        img[max(cy - 3, 0):cy + 4, max(cx - 1, 0):cx + 1] = 255   # 2px-wide cross
        img[max(cy - 1, 0):cy + 1, max(cx - 3, 0):cx + 4] = 255
        return img.transpose(2, 0, 1).copy()

    def render(self):
        return self._obs().transpose(1, 2, 0)

    def step(self, action):
        move, click = int(action[0]), int(action[1])
        before = np.abs(self.cursor - self.target_center()).sum()
        self.cursor = np.clip(self.cursor + DIRS[move] * self.step_px, [0, BAR_H], self.size - 1)
        after = np.abs(self.cursor - self.target_center()).sum()
        self.steps += 1
        reward, terminated, success = -0.01, False, False
        reward += self.shaping * (before - after) / self.size   # potential-based, optional
        if click:
            hit = self._hit()
            if hit == self.target:
                reward, terminated, success = 1.0, True, True
            else:
                reward -= 0.1
        truncated = self.steps >= self.max_steps
        return self._obs(), reward, terminated, truncated, {"success": success}


def expert_action(env: ScreenClickEnv, rng: np.random.Generator, noise=0.0):
    """Scripted demonstrator standing in for a human: head to the target, click inside it."""
    if rng.random() < noise:
        return np.array([rng.integers(len(DIRS)), int(rng.random() < 0.1)])
    d = env.target_center() - env.cursor
    if env._hit() == env.target and np.abs(d).max() <= env.step_px:
        return np.array([0, 1])
    step = np.sign(d) * (np.abs(d) > env.step_px // 2)
    move = int(np.where((DIRS == step).all(1))[0][0])
    return np.array([move, 0])
