"""Record demonstrations. A noisy scripted expert stands in for human recordings here;
real human traces (screenshot, mouse delta, click) can be saved in the same .npz format."""
import argparse
from pathlib import Path

import numpy as np

from sca.common import load_config, make_env
from sca.env import expert_action


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    env, rng = make_env(cfg), np.random.default_rng(0)
    obs_buf, act_buf, successes = [], [], 0
    for ep in range(cfg["demos"]["episodes"]):
        obs, _ = env.reset(seed=ep)
        done = False
        while not done:
            a = expert_action(env, rng, cfg["demos"]["noise"])
            obs_buf.append(obs)
            act_buf.append(a)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        successes += info["success"]
    out = Path(cfg["artifacts_dir"]) / "demos.npz"
    np.savez_compressed(out, obs=np.stack(obs_buf), actions=np.stack(act_buf))
    print(f"{len(obs_buf):,} transitions from {cfg['demos']['episodes']} demos "
          f"(demonstrator success {successes / cfg['demos']['episodes']:.1%}) -> {out}")


if __name__ == "__main__":
    main()
