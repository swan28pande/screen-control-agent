"""CNN actor-critic over screenshots, with factored action heads (move, click).

Encoder design for pointing tasks:
  * CoordConv: x/y channels so convolutions know where they are on screen
  * FiLM: the instruction (task-bar color) modulates conv features, highlighting the target
  * Spatial softmax: each feature map collapses to an expected (x, y) keypoint, a compact,
    translation-aware representation (Levine et al., 2016)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

BAR_H = 4  # instruction band height (see env.py)


class SpatialSoftmax(nn.Module):
    def forward(self, x):  # (B, C, H, W) -> (B, 2C)
        b, c, h, w = x.shape
        p = F.softmax(x.flatten(2), -1).view(b, c, h, w)
        ys = torch.linspace(-1, 1, h, device=x.device)
        xs = torch.linspace(-1, 1, w, device=x.device)
        return torch.cat([(p.sum(2) * xs).sum(-1), (p.sum(3) * ys).sum(-1)], -1)


class ScreenPolicy(nn.Module):
    def __init__(self, n_move=9, n_click=2, size=64, ch=32, keypoints=16):
        super().__init__()
        self.conv1 = nn.Conv2d(3 + 2, ch, 5, stride=2, padding=2)        # 64 -> 32
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv3 = nn.Conv2d(ch, keypoints, 3, padding=1)
        self.film = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2 * ch))
        self.keypoints = SpatialSoftmax()
        self.trunk = nn.Sequential(nn.Linear(2 * keypoints + 3, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())
        self.move_head = nn.Linear(256, n_move)
        self.click_head = nn.Linear(256, n_click)
        self.value_head = nn.Linear(256, 1)
        ys, xs = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
        self.register_buffer("coords", torch.stack([xs, ys])[None], persistent=False)

    def forward(self, obs):
        """obs: (B, 3, H, W) uint8 or float in [0, 255]."""
        x = obs.float() / 255.0
        instr = x[:, :, :BAR_H].mean((2, 3))                                # task-bar color (B, 3)
        coords = torch.zeros_like(x[:, :2]) + self.coords                   # broadcast; export-friendly
        h = F.relu(self.conv1(torch.cat([x, coords], 1)))
        gamma, beta = self.film(instr).chunk(2, -1)
        h = F.relu((1 + gamma[..., None, None]) * self.conv2(h) + beta[..., None, None])
        z = self.trunk(torch.cat([self.keypoints(self.conv3(h)), instr], -1))
        return self.move_head(z), self.click_head(z), self.value_head(z).squeeze(-1)

    def dist(self, obs):
        move, click, value = self(obs)
        return Categorical(logits=move), Categorical(logits=click), value

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        dm, dc, value = self.dist(obs)
        if deterministic:
            a = torch.stack([dm.probs.argmax(-1), dc.probs.argmax(-1)], -1)
        else:
            a = torch.stack([dm.sample(), dc.sample()], -1)
        return a, dm.log_prob(a[:, 0]) + dc.log_prob(a[:, 1]), value

    def evaluate(self, obs, actions):
        dm, dc, value = self.dist(obs)
        logp = dm.log_prob(actions[:, 0]) + dc.log_prob(actions[:, 1])
        return logp, dm.entropy() + dc.entropy(), value


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
