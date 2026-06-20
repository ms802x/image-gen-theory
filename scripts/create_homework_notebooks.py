import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").split("\n")],
    }


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "pygments_lexer": "ipython3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


COMMON_SETUP = """
import math
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F

torch.manual_seed(7)
random.seed(7)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)
"""


def write_notebook(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


def hw0():
    cells = [
        md("""
# HW0 - System Map

Goal: map a Qwen-like text-to-image system into a tiny version before writing model code.

Sources:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0 Technical Report: https://arxiv.org/abs/2605.13565
- DiT: https://arxiv.org/abs/2212.09748
- Latent Diffusion: https://arxiv.org/abs/2112.10752
"""),
        md("""
## Big Picture

Real system:

```text
prompt -> tokenizer -> text encoder -> text tokens
noise latent -> DiT/MMDiT conditioned on text -> generated latent -> VAE decoder -> image
```

Tiny system:

```text
caption -> tiny tokenizer -> tiny text encoder
noise/image latent tokens -> tiny DiT/MMDiT -> sampler -> decoder -> 16x16 or 32x32 image
```
"""),
        code("""
components = [
    ("tokenizer", "text string", "token ids"),
    ("text encoder", "token ids", "text vectors"),
    ("image representation", "pixels", "latents or patch tokens"),
    ("noise process", "clean sample + time", "noisy sample"),
    ("backbone", "noisy image tokens + text + time", "noise or velocity prediction"),
    ("sampler", "noise + model predictions", "generated latent"),
    ("decoder", "latent", "image"),
]

for name, inp, out in components:
    print(f"{name:22s}: {inp:35s} -> {out}")
"""),
        md("""
## What To Notice

- Text and images both become vectors.
- The generator does not create an image in one step. It iteratively updates noise.
- The VAE/decoder is not a detail. If reconstruction is bad, generation is capped.
- Qwen-scale systems differ mostly by data, scale, VAE quality, multimodal conditioning, and engineering.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw0_system_map/hw0_system_map.ipynb", cells)


def hw1():
    cells = [
        md("""
# HW1 - 2D Flow Matching

Question: how can a model learn to move noise into data?

Sources to inspect first:

- TorchCFM 2D tutorials: https://github.com/atong01/conditional-flow-matching/tree/main/examples/2D_tutorials
- Meta flow_matching examples: https://github.com/facebookresearch/flow_matching/tree/main/examples
- Flow Matching paper: https://arxiv.org/abs/2210.02747

This notebook intentionally uses a tiny MLP instead of a framework abstraction.
"""),
        code(COMMON_SETUP),
        code("""
def sample_moons(n, noise=0.06):
    # Small local version of two moons so the notebook has no sklearn dependency.
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1) * math.pi
    t2 = torch.rand(n2) * math.pi
    x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    x2 = torch.stack([1 - torch.cos(t2), 0.5 - torch.sin(t2)], dim=1)
    x = torch.cat([x1, x2], dim=0)
    x = x + noise * torch.randn_like(x)
    x = (x - x.mean(0)) / x.std(0)
    return x

data = sample_moons(4096)
noise = torch.randn_like(data)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1); plt.scatter(noise[:,0], noise[:,1], s=3); plt.title("source noise"); plt.axis("equal")
plt.subplot(1, 2, 2); plt.scatter(data[:,0], data[:,1], s=3); plt.title("target data"); plt.axis("equal")
plt.show()
"""),
        md("""
## Core Idea

Pick a noise point `x0` and a data point `x1`.

For the simplest straight path:

```text
x_t = (1 - t) * x0 + t * x1
velocity = x1 - x0
```

The model sees `(x_t, t)` and learns the velocity. Sampling starts from noise and follows predicted velocity.
"""),
        code("""
class TimeMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        return self.net(torch.cat([x, t], dim=1))

model = TimeMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code("""
def train_flow(steps=3000, batch_size=512):
    losses = []
    for step in range(steps):
        x1 = sample_moons(batch_size).to(device)
        x0 = torch.randn_like(x1)
        t = torch.rand(batch_size, 1, device=device)
        xt = (1 - t) * x0 + t * x1
        target_v = x1 - x0
        pred_v = model(xt, t.squeeze(1))
        loss = F.mse_loss(pred_v, target_v)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if step % 500 == 0:
            print(step, loss.item())
    return losses

losses = train_flow()
plt.plot(losses); plt.title("flow matching loss"); plt.show()
"""),
        code("""
@torch.no_grad()
def sample_flow(n=2048, steps=80, keep_trajectory=False):
    x = torch.randn(n, 2, device=device)
    traj = [x.cpu()]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device)
        v = model(x, t)
        x = x + dt * v
        if keep_trajectory and i % max(1, steps // 8) == 0:
            traj.append(x.cpu())
    return (x.cpu(), traj) if keep_trajectory else x.cpu()

samples, traj = sample_flow(keep_trajectory=True)
plt.figure(figsize=(5, 5))
plt.scatter(samples[:,0], samples[:,1], s=3)
plt.title("samples after following learned velocity")
plt.axis("equal")
plt.show()
"""),
        code("""
fig, axes = plt.subplots(1, len(traj), figsize=(3 * len(traj), 3))
for ax, points, i in zip(axes, traj, range(len(traj))):
    ax.scatter(points[:,0], points[:,1], s=2)
    ax.set_title(f"step {i}")
    ax.axis("equal")
    ax.axis("off")
plt.show()
"""),
        md("""
## Comparison Notes

- Flow matching target: velocity.
- Sampler: Euler updates through time.
- Debug signal: trajectories should smoothly move from noise into data.
- Failure mode: too few steps or too small a model gives blurry/incorrect point clouds.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw1_2d_flow/hw1_2d_flow.ipynb", cells)


def hw2():
    cells = [
        md("""
# HW2 - 2D Diffusion

Question: how is denoising different from flow?

Sources:

- DDPM: https://arxiv.org/abs/2006.11239
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
- Hugging Face Annotated Diffusion: https://github.com/huggingface/blog/blob/main/annotated-diffusion.md
"""),
        code(COMMON_SETUP),
        code("""
def sample_moons(n, noise=0.06):
    n1 = n // 2
    n2 = n - n1
    t1 = torch.rand(n1) * math.pi
    t2 = torch.rand(n2) * math.pi
    x1 = torch.stack([torch.cos(t1), torch.sin(t1)], dim=1)
    x2 = torch.stack([1 - torch.cos(t2), 0.5 - torch.sin(t2)], dim=1)
    x = torch.cat([x1, x2], dim=0)
    x = x + noise * torch.randn_like(x)
    return (x - x.mean(0)) / x.std(0)

T = 100
betas = torch.linspace(1e-4, 0.04, T)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)

def q_sample(x0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    ab = alpha_bars[t].to(x0.device)[:, None]
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps
"""),
        code("""
x0 = sample_moons(2000)
ts = [0, 10, 30, 60, 99]
fig, axes = plt.subplots(1, len(ts), figsize=(15, 3))
for ax, ti in zip(axes, ts):
    t = torch.full((x0.shape[0],), ti, dtype=torch.long)
    xt, _ = q_sample(x0, t)
    ax.scatter(xt[:,0], xt[:,1], s=2)
    ax.set_title(f"t={ti}")
    ax.axis("equal")
    ax.axis("off")
plt.show()
"""),
        code("""
class DenoiseMLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x, t):
        t = t.float() / (T - 1)
        return self.net(torch.cat([x, t[:, None]], dim=1))

model = DenoiseMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code("""
losses = []
for step in range(4000):
    x0 = sample_moons(512).to(device)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t.cpu())
    xt, eps = xt.to(device), eps.to(device)
    pred_eps = model(xt, t)
    loss = F.mse_loss(pred_eps, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if step % 500 == 0:
        print(step, loss.item())

plt.plot(losses); plt.title("DDPM noise prediction loss"); plt.show()
"""),
        code("""
@torch.no_grad()
def sample_ddpm(n=2048):
    x = torch.randn(n, 2, device=device)
    snapshots = []
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        beta = betas[ti].to(device)
        alpha = alphas[ti].to(device)
        ab = alpha_bars[ti].to(device)
        pred_eps = model(x, t)
        mean = (1 / alpha.sqrt()) * (x - beta / (1 - ab).sqrt() * pred_eps)
        if ti > 0:
            x = mean + beta.sqrt() * torch.randn_like(x)
        else:
            x = mean
        if ti in [99, 75, 50, 25, 0]:
            snapshots.append((ti, x.cpu()))
    return x.cpu(), snapshots

samples, snapshots = sample_ddpm()
plt.scatter(samples[:,0], samples[:,1], s=3)
plt.title("DDPM samples")
plt.axis("equal")
plt.show()
"""),
        md("""
## Diffusion vs Flow Matching

| Question | Flow matching | Diffusion |
| --- | --- | --- |
| Model predicts | velocity | noise, clean data, or score |
| Sampling | integrate an ODE-like path | reverse a noising chain |
| Debug visualization | trajectories | denoising snapshots |
| Main intuition | move points | clean points |
"""),
    ]
    write_notebook(ROOT / "homeworks/hw2_2d_diffusion/hw2_2d_diffusion.ipynb", cells)


def synthetic_cells(title, question, sources, body):
    return [
        md(f"""
# {title}

Question: {question}

Sources:
{sources}
"""),
        code(COMMON_SETUP),
        code("""
def make_shape_batch(batch=64, size=32):
    # Synthetic captioned shapes. This keeps labels perfect and debugging simple.
    colors = {
        "red": torch.tensor([1.0, 0.1, 0.1]),
        "green": torch.tensor([0.1, 0.8, 0.2]),
        "blue": torch.tensor([0.1, 0.25, 1.0]),
        "yellow": torch.tensor([1.0, 0.9, 0.1]),
    }
    shape_names = ["circle", "square"]
    images, captions = [], []
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size), torch.linspace(-1, 1, size), indexing="ij")
    for _ in range(batch):
        color_name = random.choice(list(colors))
        shape = random.choice(shape_names)
        color = colors[color_name][:, None, None]
        if shape == "circle":
            mask = (xx**2 + yy**2) < random.uniform(0.25, 0.45)
        else:
            w = random.uniform(0.45, 0.75)
            mask = (xx.abs() < w) & (yy.abs() < w)
        img = torch.zeros(3, size, size)
        img[:, mask] = color
        images.append(img)
        captions.append(f"{color_name} {shape}")
    return torch.stack(images), captions

images, captions = make_shape_batch(16)
fig, axes = plt.subplots(2, 8, figsize=(12, 3))
for ax, img, cap in zip(axes.flatten(), images, captions):
    ax.imshow(img.permute(1, 2, 0).clamp(0, 1))
    ax.set_title(cap, fontsize=8)
    ax.axis("off")
plt.show()
"""),
        md(body),
    ]


def hw3_to_hw11():
    specs = [
        ("hw3_tiny_images", "HW3 - Tiny Unconditional Images", "What changes when data is an image instead of a 2D point?", """
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
- DDPM: https://arxiv.org/abs/2006.11239
- lucidrains DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch
""", """
## Assignment

Train a tiny unconditional image generator on synthetic shapes or MNIST.

Minimum implementation:

- visualize image batch
- add noise at different timesteps
- train a tiny CNN/U-Net/MLP to predict noise
- sample a grid of images

Comparison to write:

- 2D points vs images
- MLP vs convolutional model
- what failure looks like when the model only learns color but not shape
"""),
        ("hw4_autoencoder_vae", "HW4 - Tiny Autoencoder / VAE", "Why do modern text-to-image systems generate latents instead of pixels?", """
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
""", """
## Assignment

Train a tiny autoencoder first, then optionally turn it into a VAE.

Minimum implementation:

- encoder: image -> latent
- decoder: latent -> image
- reconstruction loss
- original vs reconstruction grid

Comparison to write:

- autoencoder vs VAE
- latent size vs reconstruction quality
- which details disappear first
"""),
        ("hw5_latent_generation", "HW5 - Latent Diffusion Or Latent Flow", "Can we generate images by generating latents?", """
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- DDPM: https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747
""", """
## Assignment

Freeze the HW4 encoder/decoder. Train a generator in latent space.

Minimum implementation:

- encode images to latents
- train diffusion or flow on latents
- decode sampled latents
- compare with pixel-space generation

Qwen connection:

- modern systems spend most generation compute in latent space
- VAE quality can bottleneck everything
"""),
        ("hw6_text_conditioning", "HW6 - Text Conditioning", "How does text become a useful condition for generation?", """
- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
""", """
## Assignment

Use synthetic captions like `red circle` and `blue square`.

Minimum implementation:

- word-level tokenizer
- embedding table
- pooled text embedding
- concatenate/add text condition to model input
- sample the same noise with different captions

Comparison to write:

- class label conditioning vs text conditioning
- prompt ignored vs prompt followed
"""),
        ("hw7_cfg", "HW7 - Classifier-Free Guidance", "How can prompts become stronger during sampling?", """
- Classifier-Free Guidance: https://arxiv.org/abs/2207.12598
- Latent Diffusion: https://arxiv.org/abs/2112.10752
""", """
## Assignment

Add conditional dropout during training and guidance scale during sampling.

Minimum implementation:

- sometimes replace caption condition with null condition
- sample unconditional and conditional predictions
- combine them with guidance scale
- compare guidance scales visually

Key intuition:

CFG pushes samples away from generic unconditional generation toward prompt-conditioned generation.
"""),
        ("hw8_tiny_dit", "HW8 - Tiny DiT", "How does a transformer replace a U-Net?", """
- DiT: https://arxiv.org/abs/2212.09748
- Official DiT repo: https://github.com/facebookresearch/DiT
""", """
## Assignment

Patchify image latents into tokens and train a tiny transformer denoiser.

Minimum implementation:

- patchify
- timestep embedding
- transformer blocks
- prediction head
- unpatchify

Comparison to write:

- U-Net spatial bias vs transformer token mixing
- where image position information enters
"""),
        ("hw9_tiny_mmdit", "HW9 - Tiny MMDiT-Style Conditioning", "How do text tokens and image tokens interact?", """
- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- DiT: https://arxiv.org/abs/2212.09748
""", """
## Assignment

Stop using only a pooled text vector. Keep text as tokens and image as patch tokens.

Minimum implementation:

- caption tokens: `[B, T_text, D]`
- image tokens: `[B, T_img, D]`
- transformer interaction
- predict image-token noise/velocity only

Comparison to write:

- pooled text conditioning vs token-level conditioning
- why long prompts need token-level representations
"""),
        ("hw10_final_tiny_qwen", "HW10 - Final Tiny Qwen-Like Text-To-Image", "Can the pieces become one tiny end-to-end model?", """
- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
- DiT: https://arxiv.org/abs/2212.09748
- Latent Diffusion: https://arxiv.org/abs/2112.10752
""", """
## Assignment

Assemble the final tiny text-to-image model.

Minimum implementation:

- synthetic captioned dataset
- tiny tokenizer and text encoder
- tiny latent representation or autoencoder
- tiny DiT/MMDiT
- diffusion or flow objective
- classifier-free guidance
- sampler and decoder

Ablations:

- no text conditioning
- pooled text conditioning
- token-level text conditioning
- low vs high guidance

Pass condition:

Simple prompts change generated samples in the expected direction.
"""),
        ("hw11_qwen_mapping", "HW11 - Qwen Paper Mapping", "Which parts are architecture, data, scale, and engineering?", """
- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
- Qwen-Image-2.0 Technical Report: https://arxiv.org/abs/2605.10730
""", """
## Assignment

Map your tiny implementation back to Qwen-like systems.

Make a table:

| Qwen component | Tiny version | What was learned | What remains unknown |
| --- | --- | --- | --- |

Do not treat the paper as magic. Separate:

- architecture ideas
- data quality
- scale
- training curriculum
- evaluation
- deployment engineering
"""),
    ]
    for i, (slug, title, question, sources, body) in enumerate(specs, start=3):
        cells = synthetic_cells(title, question, sources, body)
        cells.append(code("""
# Scratch cell for this homework.
# Replace this with the implementation after surveying the listed sources.
print("Next: implement this homework after source inspection.")
"""))
        write_notebook(ROOT / f"homeworks/{slug}/{slug}.ipynb", cells)


def readme():
    path = ROOT / "homeworks/README.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("""# Homeworks

This folder contains the notebook sequence from `ROADMAP.md`.

The notebooks are intentionally small and source-grounded. Early notebooks are more complete because they teach the core mechanics. Later notebooks are structured lab notebooks that should be filled in after inspecting the linked public sources.

Run order:

1. `hw0_system_map/hw0_system_map.ipynb`
2. `hw1_2d_flow/hw1_2d_flow.ipynb`
3. `hw2_2d_diffusion/hw2_2d_diffusion.ipynb`
4. `hw3_tiny_images/hw3_tiny_images.ipynb`
5. `hw4_autoencoder_vae/hw4_autoencoder_vae.ipynb`
6. `hw5_latent_generation/hw5_latent_generation.ipynb`
7. `hw6_text_conditioning/hw6_text_conditioning.ipynb`
8. `hw7_cfg/hw7_cfg.ipynb`
9. `hw8_tiny_dit/hw8_tiny_dit.ipynb`
10. `hw9_tiny_mmdit/hw9_tiny_mmdit.ipynb`
11. `hw10_final_tiny_qwen/hw10_final_tiny_qwen.ipynb`
12. `hw11_qwen_mapping/hw11_qwen_mapping.ipynb`
""", encoding="utf-8")


def main():
    hw0()
    hw1()
    hw2()
    hw3_to_hw11()
    readme()


if __name__ == "__main__":
    main()
