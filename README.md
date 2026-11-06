# Screen Control Policy Agent

A GUI control agent that maps screenshots to mouse actions. It learns first by behavioral cloning (BC) from demonstrations, then gets fine-tuned with PPO, and is exported to INT8 ONNX for edge inference.

**Stack:** Python, PyTorch (CUDA when available), Gymnasium, ONNX, ONNX Runtime (CUDA or CPU execution providers).

## Task
`sca/env.py` is a synthetic 64×64 desktop. The colored bar at the top is the instruction: click the button with that color.
- **Observations:** raw RGB screenshots.
- **Actions:** `MultiDiscrete([9, 2])`, a cursor move (stay or one of 8 directions) plus click / no click.
- **Rewards:** +1 for clicking the right button, −0.1 for a wrong click, −0.01 per step.

## Pipeline
```bash
pip install -r requirements.txt
python -m sca.collect_demos   # demonstrations → artifacts/demos.npz
python -m sca.train_bc        # behavioral cloning → artifacts/bc.pt
python -m sca.train_ppo       # PPO fine-tune from BC (critic warm-up first) → artifacts/ppo.pt
python -m sca.export_onnx     # ONNX FP32 + static INT8 (QDQ, calibrated on demo frames); compares success + latency
python -m sca.live --model artifacts/policy_int8.onnx   # experimental desktop harness, dry run by default
```
- `collect_demos` uses a noisy scripted expert in place of human recordings. Real human traces (screenshot, move, click) drop into the same `.npz` format.
- `train_ppo --from-scratch` runs the no-BC baseline.

## Results (CPU, default config)
| Stage | Success | Steps / episode |
|---|---|---|
| Scripted demonstrator (15% random actions) | 100% | 10.8 |
| Behavioral cloning | 98% | 10.1 |
| + PPO fine-tune | 99–100% | 9.4 |

| Backend | Model size | Success | Latency (batch 1) |
|---|---|---|---|
| PyTorch | – | 100% | 0.72 ms |
| ONNX FP32 | 439 KB | 100% | 0.60 ms |
| ONNX INT8 (static QDQ) | 195 KB | 98% | 0.64 ms |

INT8 cuts model size 2.25× for a small accuracy cost. On this laptop CPU it isn't faster; the speed-up shows on INT8-capable accelerators (TensorRT, NPUs).

The encoder uses CoordConv, FiLM conditioning on the instruction color and a spatial-softmax keypoint head. A plain conv-and-flatten CNN memorized frames and only reached about 5% success with the same data.

## Layout
```
sca/env.py           Gymnasium env + scripted demonstrator
sca/policy.py        CNN actor-critic, factored move/click heads
sca/train_bc.py      behavioral cloning
sca/train_ppo.py     PPO (GAE, clipped objective, critic warm-up to protect the BC policy)
sca/export_onnx.py   ONNX export, INT8 quantization, torch vs ONNX benchmark
sca/live.py          screen capture → ONNX Runtime → mouse (mss + pyautogui)
```

## Status
Early skeleton. Next:
- A real demo recorder (screen plus mouse hooks).
- Richer synthetic UIs (text, menus, scrolling).
- Instruction text through a language encoder.
- TensorRT execution provider.
