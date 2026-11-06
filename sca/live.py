"""Experimental: drive the real desktop with the ONNX policy.

Captures a screen region, resizes it to the policy's input, runs ONNX Runtime and moves
or clicks the mouse. The policy is trained on the synthetic env, so this is a
deployment harness to adapt, not a general-purpose agent. Defaults to --dry-run.

    pip install mss pyautogui
    python -m sca.live --model artifacts/policy_int8.onnx --region 0 0 640 640 [--execute]
"""
import argparse
import time

import numpy as np
import onnxruntime as ort

from sca.env import DIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="artifacts/policy_int8.onnx")
    ap.add_argument("--region", type=int, nargs=4, default=[0, 0, 640, 640], metavar=("LEFT", "TOP", "W", "H"))
    ap.add_argument("--hz", type=float, default=10)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--execute", action="store_true", help="actually move/click (default: dry run)")
    args = ap.parse_args()

    import mss
    import pyautogui

    sess = ort.InferenceSession(args.model, providers=ort.get_available_providers())
    size = sess.get_inputs()[0].shape[-1]
    left, top, w, h = args.region
    scale = np.array([w, h]) / size
    with mss.mss() as sct:
        for _ in range(args.steps):
            t0 = time.perf_counter()
            frame = np.asarray(sct.grab({"left": left, "top": top, "width": w, "height": h}))[..., :3][..., ::-1]
            ys, xs = np.linspace(0, h - 1, size).astype(int), np.linspace(0, w - 1, size).astype(int)
            obs = frame[ys][:, xs].transpose(2, 0, 1)[None].astype(np.float32)
            move, click = sess.run(None, {"obs": obs})
            dx, dy = DIRS[move[0].argmax()] * 3 * scale
            do_click = bool(click[0].argmax())
            ms = (time.perf_counter() - t0) * 1e3
            print(f"move ({dx:+.0f},{dy:+.0f}) click={do_click} [{ms:.1f} ms]")
            if args.execute:
                pyautogui.moveRel(dx, dy, _pause=False)
                if do_click:
                    pyautogui.click(_pause=False)
            time.sleep(max(0.0, 1 / args.hz - (time.perf_counter() - t0)))


if __name__ == "__main__":
    main()
