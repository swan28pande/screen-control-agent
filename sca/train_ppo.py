"""PPO fine-tuning starting from the BC policy."""
import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from sca.common import evaluate, load_config, make_env, torch_actor
from sca.policy import ScreenPolicy, device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--from-scratch", action="store_true", help="skip BC init (baseline)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    art, c, dev = Path(cfg["artifacts_dir"]), cfg["ppo"], device()

    model = ScreenPolicy(size=cfg["env"]["size"]).to(dev)
    if not args.from_scratch:
        model.load_state_dict(torch.load(art / "bc.pt", map_location=dev))
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"], eps=1e-5)
    envs = gym.vector.SyncVectorEnv([lambda i=i: make_env(cfg) for i in range(c["n_envs"])])
    obs, _ = envs.reset(seed=123)
    N, S = c["n_envs"], c["rollout_steps"]
    sr0, _ = evaluate(torch_actor(model, dev), cfg, episodes=100)
    print(f"start success {sr0:.1%} ({'scratch' if args.from_scratch else 'BC init'}), device={dev}")

    best = -1.0
    for it in range(c["iterations"]):
        buf = {k: [] for k in ("obs", "act", "logp", "val", "rew", "done")}
        for _ in range(S):
            o = torch.as_tensor(obs, device=dev)
            a, logp, v = model.act(o)
            obs, r, term, trunc, _ = envs.step(a.cpu().numpy())
            for k, x in zip(buf, (o, a, logp, v, torch.as_tensor(r, dtype=torch.float32, device=dev),
                                  torch.as_tensor(term | trunc, dtype=torch.float32, device=dev))):
                buf[k].append(x)
        b = {k: torch.stack(v) for k, v in buf.items()}            # (S, N, ...)
        with torch.no_grad():
            _, _, next_v = model(torch.as_tensor(obs, device=dev))
        # GAE (truncation treated as termination for simplicity)
        adv = torch.zeros(S, N, device=dev)
        last = torch.zeros(N, device=dev)
        for t in reversed(range(S)):
            nv = next_v if t == S - 1 else b["val"][t + 1]
            nonterm = 1 - b["done"][t]
            delta = b["rew"][t] + c["gamma"] * nv * nonterm - b["val"][t]
            last = delta + c["gamma"] * c["gae_lambda"] * nonterm * last
            adv[t] = last
        ret = adv + b["val"]
        flat = lambda x: x.reshape(S * N, *x.shape[2:])
        fo, fa, flp, fadv, fret = map(flat, (b["obs"], b["act"], b["logp"], adv, ret))
        fadv = (fadv - fadv.mean()) / (fadv.std() + 1e-8)
        critic_only = it < c["value_warmup_iters"]

        for _ in range(c["epochs"]):
            for idx in torch.randperm(S * N, device=dev).split(c["minibatch"]):
                logp, ent, v = model.evaluate(fo[idx], fa[idx])
                v_loss = ((v - fret[idx]) ** 2).mean()
                if critic_only:
                    loss = c["vf_coef"] * v_loss
                else:
                    ratio = (logp - flp[idx]).exp()
                    pg = -torch.min(ratio * fadv[idx], ratio.clamp(1 - c["clip"], 1 + c["clip"]) * fadv[idx]).mean()
                    loss = pg + c["vf_coef"] * v_loss - c["ent_coef"] * ent.mean()
                opt.zero_grad(); loss.backward()
                if critic_only:  # only the value head learns during warmup; the BC policy stays intact
                    for name, p in model.named_parameters():
                        if not name.startswith("value_head"):
                            p.grad = None
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                opt.step()

        if it % 10 == 9 or it == c["iterations"] - 1:
            sr, steps = evaluate(torch_actor(model, dev), cfg, episodes=100)
            print(f"[ppo] iter {it + 1:3d}  rollout reward/step {b['rew'].mean():+.4f}  eval success {sr:.1%}  steps {steps:.1f}")
            if sr >= best:
                best = sr
                torch.save(model.state_dict(), art / "ppo.pt")
    print(f"best eval success {best:.1%} -> {art / 'ppo.pt'}")


if __name__ == "__main__":
    main()
