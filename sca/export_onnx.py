"""Export the policy to ONNX, INT8-quantize it (static, QDQ, calibrated on demo frames),
and compare accuracy/latency of PyTorch vs ONNX FP32 vs ONNX INT8."""
import argparse
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, QuantFormat, QuantType, quantize_static

from sca.common import evaluate, load_config, torch_actor
from sca.policy import ScreenPolicy, device


class Actor(torch.nn.Module):
    """Inference graph: screenshot (float32 0-255) -> (move_logits, click_logits)."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, obs):
        move, click, _ = self.policy(obs)
        return move, click


class DemoFrames(CalibrationDataReader):
    def __init__(self, frames, batch=32):
        self.it = iter([{"obs": frames[i:i + batch].astype(np.float32)} for i in range(0, len(frames), batch)])

    def get_next(self):
        return next(self.it, None)


def providers():
    avail = ort.get_available_providers()
    return [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail]


def onnx_actor(sess):
    def act(obs):
        move, click = sess.run(None, {"obs": obs[None].astype(np.float32)})
        return np.array([move[0].argmax(), click[0].argmax()])
    return act


def latency_ms(fn, obs, n=300):
    for _ in range(20):
        fn(obs)
    t0 = time.perf_counter()
    for _ in range(n):
        fn(obs)
    return (time.perf_counter() - t0) / n * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", default=None, help="defaults to ppo.pt, falling back to bc.pt")
    args = ap.parse_args()
    cfg = load_config(args.config)
    art = Path(cfg["artifacts_dir"])
    ckpt = Path(args.checkpoint) if args.checkpoint else (art / "ppo.pt" if (art / "ppo.pt").exists() else art / "bc.pt")

    size = cfg["env"]["size"]
    policy = ScreenPolicy(size=size)
    policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
    policy.eval()

    fp32, int8 = art / "policy_fp32.onnx", art / "policy_int8.onnx"
    torch.onnx.export(Actor(policy), (torch.zeros(1, 3, size, size),), str(fp32), input_names=["obs"],
                      output_names=["move_logits", "click_logits"], dynamic_axes={"obs": {0: "batch"}},
                      opset_version=17, dynamo=False)
    frames = np.load(art / "demos.npz")["obs"]
    calib = frames[np.random.default_rng(0).choice(len(frames), 512, replace=False)]
    quantize_static(str(fp32), str(int8), DemoFrames(calib), quant_format=QuantFormat.QDQ,
                    activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8, per_channel=True)

    prov = providers()
    sessions = {name: ort.InferenceSession(str(p), providers=prov) for name, p in [("onnx-fp32", fp32), ("onnx-int8", int8)]}
    dev = device()
    policy_dev = policy.to(dev)
    actors = {"torch": torch_actor(policy_dev, dev), **{k: onnx_actor(s) for k, s in sessions.items()}}
    sample = frames[0]
    print(f"checkpoint {ckpt.name}; onnxruntime providers {prov}")
    print(f"{'backend':10s} {'size':>9s} {'success':>8s} {'latency ms':>10s}")
    for name, act in actors.items():
        sr, _ = evaluate(act, cfg, episodes=cfg["eval_episodes"])
        size_kb = {"onnx-fp32": fp32, "onnx-int8": int8}.get(name)
        size_s = f"{size_kb.stat().st_size / 1024:.0f} KB" if size_kb else "-"
        print(f"{name:10s} {size_s:>9s} {sr:8.1%} {latency_ms(act, sample):10.3f}")


if __name__ == "__main__":
    main()
