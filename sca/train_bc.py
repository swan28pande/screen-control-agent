"""Behavioral cloning: maximize log-likelihood of demonstrated actions."""
import argparse
from pathlib import Path

import numpy as np
import torch

from sca.common import evaluate, load_config, torch_actor
from sca.policy import ScreenPolicy, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    cfg = load_config(ap.parse_args().config)
    art, c, dev = Path(cfg["artifacts_dir"]), cfg["bc"], device()
    d = np.load(art / "demos.npz")
    obs, act = torch.from_numpy(d["obs"]), torch.from_numpy(d["actions"]).long()
    model = ScreenPolicy(size=cfg["env"]["size"]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    print(f"BC on {len(obs):,} transitions, device={dev}")
    for ep in range(c["epochs"]):
        perm, tot = torch.randperm(len(obs)), 0.0
        for i in range(0, len(obs), c["batch_size"]):
            b = perm[i:i + c["batch_size"]]
            logp, _, _ = model.evaluate(obs[b].to(dev), act[b].to(dev))
            loss = -logp.mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(b)
        model.eval()
        sr, steps = evaluate(torch_actor(model, dev), cfg, episodes=100)
        model.train()
        print(f"[bc] epoch {ep} nll {tot / len(obs):.4f}  success {sr:.1%}  steps {steps:.1f}")
    torch.save(model.state_dict(), art / "bc.pt")


if __name__ == "__main__":
    main()
