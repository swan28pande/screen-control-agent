from pathlib import Path

import numpy as np
import torch
import yaml

from sca.env import ScreenClickEnv


def load_config(path="configs/default.yaml"):
    cfg = yaml.safe_load(open(path))
    Path(cfg["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
    return cfg


def make_env(cfg, **kw):
    return ScreenClickEnv(**cfg["env"], **kw)


def evaluate(act_fn, cfg, episodes=100, seed=10_000):
    """act_fn: (obs uint8 (3,H,W)) -> action array. Returns success rate and mean steps."""
    env = make_env(cfg)
    succ, steps = 0, []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs, _, term, trunc, info = env.step(act_fn(obs))
            done = term or trunc
        succ += info["success"]
        steps.append(env.steps)
    return succ / episodes, float(np.mean(steps))


def torch_actor(model, dev):
    def act(obs):
        a, _, _ = model.act(torch.as_tensor(obs[None], device=dev), deterministic=True)
        return a[0].cpu().numpy()
    return act
