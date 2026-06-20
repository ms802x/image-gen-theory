import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip("\n").split("\n")],
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


def write_notebook(path, cells):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook(cells), indent=2), encoding="utf-8")


COMMON_SETUP = r"""
import math
import random
from pathlib import Path
from io import BytesIO

import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from IPython.display import Image, display

torch.manual_seed(7)
random.seed(7)
torch.set_num_threads(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


def show_plot():
    fig = plt.gcf()
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    display(Image(data=buf.getvalue()))
    plt.close(fig)
"""

GPU_NOTE = """
> CPU smoke run: this notebook is designed to execute quickly on CPU to verify mechanics, tensor shapes, plotting, and sampler logic. Generated samples may look like noise because the default training loop is intentionally tiny. Treat noisy samples as undertraining, not as evidence that the method is wrong. For meaningful visual quality, run longer on a GPU by increasing training steps, batch size, diffusion steps, and model width.
"""


SHAPES = r"""
COLORS = {
    "red": torch.tensor([1.0, 0.08, 0.08]),
    "green": torch.tensor([0.08, 0.80, 0.20]),
    "blue": torch.tensor([0.08, 0.25, 1.0]),
    "yellow": torch.tensor([1.0, 0.88, 0.08]),
}
SHAPES = ["circle", "square", "triangle"]


def draw_triangle(xx, yy, scale):
    # Upright triangle mask in normalized coordinates.
    top = yy > -scale
    left = yy < 2 * scale * (xx + scale)
    right = yy < -2 * scale * (xx - scale)
    return top & left & right & (yy < scale)


def make_shape_batch(batch=64, size=32, with_text=True):
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, size),
        torch.linspace(-1, 1, size),
        indexing="ij",
    )
    images, captions = [], []
    for _ in range(batch):
        color_name = random.choice(list(COLORS))
        shape_name = random.choice(SHAPES)
        color = COLORS[color_name][:, None]
        scale = random.uniform(0.42, 0.72)
        img = torch.zeros(3, size, size)
        if shape_name == "circle":
            mask = xx.square() + yy.square() < scale**2
        elif shape_name == "square":
            mask = (xx.abs() < scale) & (yy.abs() < scale)
        else:
            mask = draw_triangle(xx, yy, scale)
        img[:, mask] = color
        images.append(img)
        captions.append(f"{color_name} {shape_name}")
    images = torch.stack(images)
    return (images, captions) if with_text else images


def show_images(images, titles=None, n=16, size=2.0):
    images = images.detach().cpu().clamp(0, 1)
    n = min(n, images.shape[0])
    cols = min(8, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size))
    axes = [axes] if n == 1 else axes.reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i].permute(1, 2, 0))
            if titles:
                ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    show_plot()
"""


DIFFUSION = r"""
T = 8
betas = torch.linspace(1e-4, 0.05, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    ab = alpha_bars[t].view(-1, *([1] * (x0.ndim - 1)))
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps


@torch.no_grad()
def ddpm_sample(model, shape, cond=None, guidance_scale=None, cond_fn=None):
    x = torch.randn(shape, device=device)
    for ti in reversed(range(T)):
        t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
        if guidance_scale is None:
            pred_eps = model(x, t) if cond is None else model(x, t, cond)
        else:
            eps_uncond = cond_fn(x, t, None)
            eps_cond = cond_fn(x, t, cond)
            pred_eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
        beta, alpha, ab = betas[ti], alphas[ti], alpha_bars[ti]
        mean = (1 / alpha.sqrt()) * (x - beta / (1 - ab).sqrt() * pred_eps)
        x = mean if ti == 0 else mean + beta.sqrt() * torch.randn_like(x)
    return x
"""


TEXT_UTILS = r"""
VOCAB = ["<pad>", "<null>", "red", "green", "blue", "yellow", "circle", "square", "triangle"]
stoi = {tok: i for i, tok in enumerate(VOCAB)}
itos = {i: tok for tok, i in stoi.items()}


def tokenize(captions, max_len=2, drop_prob=0.0):
    ids = []
    for cap in captions:
        if random.random() < drop_prob:
            words = ["<null>"]
        else:
            words = cap.split()
        row = [stoi.get(w, 0) for w in words[:max_len]]
        row += [0] * (max_len - len(row))
        ids.append(row)
    return torch.tensor(ids, dtype=torch.long, device=device)
"""


def hw0():
    cells = [
        md(r"""
# HW0 - System Map

Goal: map a Qwen-like text-to-image system into a tiny version before writing model code.

Grounding sources:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
- DiT: https://arxiv.org/abs/2212.09748
- Latent Diffusion: https://arxiv.org/abs/2112.10752
"""),
        md(r"""
## Architecture Map

```text
text prompt
  -> tokenizer
  -> text encoder
  -> text tokens

random latent/image noise
  -> timestep embedding
  -> image patch tokens
  -> DiT or MMDiT conditioned on text
  -> predicted noise or velocity
  -> sampler loop
  -> generated latent
  -> VAE decoder
  -> image
```

The tiny repo keeps the same conceptual interfaces but removes scale: tiny captions, synthetic images, small transformers, and short sampling loops.
"""),
        code(r"""
components = [
    ("tokenizer", "prompt string", "token ids"),
    ("text encoder", "token ids", "text vectors or text tokens"),
    ("VAE/latent encoder", "image", "compressed latent"),
    ("noise process", "clean latent + t", "noisy latent"),
    ("DiT/MMDiT", "noisy latent tokens + text + t", "noise or velocity prediction"),
    ("sampler", "noise + repeated model calls", "generated latent"),
    ("decoder", "generated latent", "image"),
]

for name, inp, out in components:
    print(f"{name:18s}: {inp:34s} -> {out}")
"""),
        md(r"""
## Interview Takeaway

You should be able to say: Qwen-Image is not just "diffusion plus text." It is a latent generative system whose quality depends on the condition encoder, VAE reconstruction and diffusability, DiT/MMDiT token mixing, sampler behavior, data curation, and training curriculum.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw0_system_map/hw0_system_map.ipynb", cells)


def hw1():
    cells = [
        md(r"""
# HW1 - 2D Flow Matching

Question: how can a model learn to move noise into data?

""" + GPU_NOTE + r"""

Sources:

- TorchCFM 2D tutorials: https://github.com/atong01/conditional-flow-matching/tree/main/examples/2D_tutorials
- Meta flow_matching examples: https://github.com/facebookresearch/flow_matching/tree/main/examples
- Flow Matching: https://arxiv.org/abs/2210.02747

This notebook uses a tiny MLP so the core idea is visible without framework abstractions.
"""),
        code(COMMON_SETUP),
        code(r"""
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


data = sample_moons(4096)
noise = torch.randn_like(data)
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, pts, title in [(axes[0], noise, "source noise"), (axes[1], data, "target data")]:
    ax.scatter(pts[:, 0], pts[:, 1], s=3)
    ax.set_title(title)
    ax.axis("equal")
show_plot()
"""),
        md(r"""
## Training Target

For the simplest straight path:

```text
x_t = (1 - t) * x0 + t * x1
target velocity = x1 - x0
```

The network sees `(x_t, t)` and predicts velocity. Sampling starts from noise and repeatedly moves in the predicted direction.
"""),
        code(r"""
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
print(sum(p.numel() for p in model.parameters()), "parameters")
"""),
        code(r"""
losses = []
for step in range(80):
    x1 = sample_moons(128).to(device)
    x0 = torch.randn_like(x1)
    t = torch.rand(x1.shape[0], 1, device=device)
    xt = (1 - t) * x0 + t * x1
    target_v = x1 - x0
    pred_v = model(xt, t.squeeze(1))
    loss = F.mse_loss(pred_v, target_v)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 20 == 0:
        print(step, round(loss.item(), 4))

plt.plot(losses)
plt.title("flow matching loss")
show_plot()
"""),
        code(r"""
@torch.no_grad()
def sample_flow(n=512, steps=24, keep_trajectory=True):
    x = torch.randn(n, 2, device=device)
    trajectory = [(0, x.cpu())]
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.full((n,), i / steps, device=device)
        x = x + dt * model(x, t)
        if keep_trajectory and i in [5, 11, 17, 23]:
            trajectory.append((i + 1, x.cpu()))
    return x.cpu(), trajectory


samples, trajectory = sample_flow()
fig, axes = plt.subplots(1, len(trajectory), figsize=(3 * len(trajectory), 3))
for ax, (step, pts) in zip(axes, trajectory):
    ax.scatter(pts[:, 0], pts[:, 1], s=2)
    ax.set_title(f"step {step}")
    ax.axis("equal")
    ax.axis("off")
show_plot()
"""),
        code(r"""
@torch.no_grad()
def plot_velocity_field(t_value=0.5):
    grid_x, grid_y = torch.meshgrid(torch.linspace(-3, 3, 20), torch.linspace(-3, 3, 20), indexing="xy")
    pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1).to(device)
    t = torch.full((pts.shape[0],), t_value, device=device)
    v = model(pts, t).cpu()
    plt.figure(figsize=(5, 5))
    plt.quiver(pts.cpu()[:, 0], pts.cpu()[:, 1], v[:, 0], v[:, 1], angles="xy")
    plt.title(f"learned velocity field at t={t_value}")
    plt.axis("equal")
    show_plot()


plot_velocity_field(0.5)
"""),
        md(r"""
## What To Say In An Interview

- Flow matching trains a vector field, not a denoiser.
- Sampling is an integration problem: start from source noise and follow the learned velocity.
- The Qwen connection is conceptual: modern systems may use flow-style objectives or diffusion-style objectives, but in both cases the model repeatedly predicts an update in latent/token space.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw1_2d_flow/hw1_2d_flow.ipynb", cells)


def hw2():
    cells = [
        md(r"""
# HW2 - 2D Diffusion

Question: how is denoising different from flow?

""" + GPU_NOTE + r"""

Sources:

- DDPM: https://arxiv.org/abs/2006.11239
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
- Hugging Face Annotated Diffusion: https://github.com/huggingface/blog/blob/main/annotated-diffusion.md
"""),
        code(COMMON_SETUP),
        code(r"""
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


T = 24
betas = torch.linspace(1e-4, 0.04, T, device=device)
alphas = 1.0 - betas
alpha_bars = torch.cumprod(alphas, dim=0)


def q_sample(x0, t, eps=None):
    if eps is None:
        eps = torch.randn_like(x0)
    ab = alpha_bars[t].view(-1, 1)
    return ab.sqrt() * x0 + (1 - ab).sqrt() * eps, eps
"""),
        code(r"""
x0 = sample_moons(2500).to(device)
timesteps = [0, 4, 8, 16, 23]
fig, axes = plt.subplots(1, len(timesteps), figsize=(15, 3))
for ax, ti in zip(axes, timesteps):
    t = torch.full((x0.shape[0],), ti, device=device, dtype=torch.long)
    xt, _ = q_sample(x0, t)
    pts = xt.cpu()
    ax.scatter(pts[:, 0], pts[:, 1], s=2)
    ax.set_title(f"t={ti}")
    ax.axis("equal")
    ax.axis("off")
show_plot()
"""),
        code(r"""
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
        t_scaled = t.float()[:, None] / (T - 1)
        return self.net(torch.cat([x, t_scaled], dim=1))


model = DenoiseMLP().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
losses = []
for step in range(100):
    x0 = sample_moons(128).to(device)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred_eps = model(xt, t)
    loss = F.mse_loss(pred_eps, eps)
    opt.zero_grad()
    loss.backward()
    opt.step()
    losses.append(loss.item())
    if step % 25 == 0:
        print(step, round(loss.item(), 4))

plt.plot(losses)
plt.title("DDPM noise prediction loss")
show_plot()
"""),
        code(r"""
@torch.no_grad()
def sample_ddpm(n=512):
    x = torch.randn(n, 2, device=device)
    snapshots = []
    for ti in reversed(range(T)):
        t = torch.full((n,), ti, device=device, dtype=torch.long)
        beta, alpha, ab = betas[ti], alphas[ti], alpha_bars[ti]
        pred_eps = model(x, t)
        mean = (1 / alpha.sqrt()) * (x - beta / (1 - ab).sqrt() * pred_eps)
        x = mean if ti == 0 else mean + beta.sqrt() * torch.randn_like(x)
        if ti in [23, 16, 8, 4, 0]:
            snapshots.append((ti, x.cpu()))
    return x.cpu(), snapshots


samples, snapshots = sample_ddpm()
fig, axes = plt.subplots(1, len(snapshots), figsize=(15, 3))
for ax, (ti, pts) in zip(axes, snapshots):
    ax.scatter(pts[:, 0], pts[:, 1], s=2)
    ax.set_title(f"t={ti}")
    ax.axis("equal")
    ax.axis("off")
show_plot()
"""),
        md(r"""
## Diffusion vs Flow

| Topic | Flow matching | Diffusion |
| --- | --- | --- |
| Target | velocity | usually noise |
| Sampling | integrate learned vector field | reverse a noising chain |
| Easy visualization | trajectories | denoising snapshots |
| Failure mode | wrong motion field | bad noise estimate at some timesteps |

Interview line: diffusion is easier to introduce as "learn to remove noise"; flow matching is easier to view as "learn the transport direction."
"""),
    ]
    write_notebook(ROOT / "homeworks/hw2_2d_diffusion/hw2_2d_diffusion.ipynb", cells)


def hw3():
    cells = [
        md(r"""
# HW3 - Tiny Unconditional Images

Question: what changes when the data is an image instead of a 2D point?

""" + GPU_NOTE + r"""

Sources:

- DDPM: https://arxiv.org/abs/2006.11239
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
- lucidrains DDPM: https://github.com/lucidrains/denoising-diffusion-pytorch
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(DIFFUSION),
        code(r"""
images = make_shape_batch(8, with_text=False)
show_images(images, n=8)

noisy_steps = [0, 1, 2, 4, 6, 7]
noisy, _ = q_sample(images[:6].to(device), torch.tensor(noisy_steps, device=device))
show_images(noisy, titles=[f"t={t}" for t in noisy_steps], n=6)
"""),
        code(r"""
class TinyImageDenoiser(nn.Module):
    def __init__(self, channels=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x, t):
        t_img = (t.float() / (T - 1)).view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_img], dim=1))


model = TinyImageDenoiser().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
losses = []
for step in range(10):
    x0 = make_shape_batch(8, with_text=False).to(device)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred = model(xt, t)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
    if step % 5 == 0:
        print(step, round(loss.item(), 4))

plt.plot(losses)
plt.title("tiny image diffusion loss")
show_plot()
"""),
        code(r"""
samples = ddpm_sample(model, (4, 3, 32, 32)).cpu()
show_images(samples, n=4)
"""),
        md(r"""
## Interview Takeaway

Images are just high-dimensional points, but the model needs spatial bias. A plain MLP can work for tiny images, but convolutions or patch tokens make the image structure easier to learn.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw3_tiny_images/hw3_tiny_images.ipynb", cells)


def hw4():
    cells = [
        md(r"""
# HW4 - Tiny Autoencoder / VAE

Question: why do modern text-to-image models generate latents instead of pixels?

""" + GPU_NOTE + r"""

Sources:

- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(r"""
class TinyVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.SiLU(),
            nn.Flatten(),
        )
        self.to_mu = nn.Linear(64 * 8 * 8, latent_dim)
        self.to_logvar = nn.Linear(64 * 8 * 8, latent_dim)
        self.from_z = nn.Linear(latent_dim, 64 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.to_mu(h), self.to_logvar(h).clamp(-8, 8)

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.from_z(z))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


vae = TinyVAE(latent_dim=64).to(device)
opt = torch.optim.AdamW(vae.parameters(), lr=2e-3)
"""),
        code(r"""
losses = []
for step in range(20):
    x = make_shape_batch(8, with_text=False).to(device)
    recon, mu, logvar = vae(x)
    recon_loss = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    loss = recon_loss + 0.001 * kl
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append((recon_loss.item(), kl.item()))
    if step % 10 == 0:
        print(step, "recon", round(recon_loss.item(), 4), "kl", round(kl.item(), 4))

plt.plot([x for x, _ in losses], label="recon")
plt.plot([k for _, k in losses], label="kl")
plt.legend()
plt.title("VAE training")
show_plot()
"""),
        code(r"""
test = make_shape_batch(8, with_text=False).to(device)
with torch.no_grad():
    recon, mu, logvar = vae(test)
pair = torch.stack([test[:4].cpu(), recon[:4].cpu()], dim=1).reshape(-1, 3, 32, 32)
titles = sum(([f"orig {i}", f"recon {i}"] for i in range(4)), [])
show_images(pair, titles=titles, n=8)
print("latent shape:", mu.shape)
"""),
        md(r"""
## What To Look For

- Blurry edges mean the latent bottleneck is losing spatial detail.
- Color mistakes mean semantics are not reliably preserved.
- Qwen-Image-VAE-2.0 emphasizes both reconstruction fidelity and diffusability: the latent must reconstruct well and also be easy for a generator to model.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw4_autoencoder_vae/hw4_autoencoder_vae.ipynb", cells)


def hw5():
    cells = [
        md(r"""
# HW5 - Latent Diffusion Or Latent Flow

Question: can we generate images by generating latents?

""" + GPU_NOTE + r"""

Sources:

- Latent Diffusion: https://arxiv.org/abs/2112.10752
- DDPM: https://arxiv.org/abs/2006.11239
- Flow Matching: https://arxiv.org/abs/2210.02747
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(DIFFUSION),
        md("This notebook trains a tiny VAE and then a small DDPM in its latent space."),
        code(r"""
class TinyAE(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.SiLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, latent_dim),
        )
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.SiLU(),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        return self.enc(x)

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.decode(self.encode(x))


ae = TinyAE().to(device)
opt_ae = torch.optim.AdamW(ae.parameters(), lr=2e-3)
for step in range(20):
    x = make_shape_batch(8, with_text=False).to(device)
    recon = ae(x)
    loss = F.mse_loss(recon, x)
    opt_ae.zero_grad(); loss.backward(); opt_ae.step()
    if step % 10 == 0:
        print("ae", step, round(loss.item(), 4))
"""),
        code(r"""
class LatentDenoiser(nn.Module):
    def __init__(self, latent_dim=64, hidden=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z, t):
        t_scaled = t.float()[:, None] / (T - 1)
        return self.net(torch.cat([z, t_scaled], dim=1))


denoiser = LatentDenoiser().to(device)
opt = torch.optim.AdamW(denoiser.parameters(), lr=2e-3)

for step in range(30):
    x = make_shape_batch(8, with_text=False).to(device)
    with torch.no_grad():
        z0 = ae.encode(x)
    t = torch.randint(0, T, (z0.shape[0],), device=device)
    zt, eps = q_sample(z0, t)
    pred = denoiser(zt, t)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 10 == 0:
        print("latent ddpm", step, round(loss.item(), 4))
"""),
        code(r"""
with torch.no_grad():
    z = ddpm_sample(denoiser, (4, 64))
    imgs = ae.decode(z).cpu()
show_images(imgs, n=4)
"""),
        md(r"""
## Interview Takeaway

Latent generation reduces the dimensionality the generator must model. The tradeoff is that all reconstruction errors and latent-space pathologies become generator errors later.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw5_latent_generation/hw5_latent_generation.ipynb", cells)


def hw6():
    cells = [
        md(r"""
# HW6 - Text Conditioning

Question: how does text become a useful condition for generation?

""" + GPU_NOTE + r"""

Sources:

- Latent Diffusion: https://arxiv.org/abs/2112.10752
- Diffusion Explainer: https://poloclub.github.io/diffusion-explainer/
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(TEXT_UTILS),
        code(DIFFUSION),
        code(r"""
images, captions = make_shape_batch(16)
show_images(images, captions, n=16)
print(captions[:6])
print(tokenize(captions[:6]))
"""),
        code(r"""
class TextConditionedDenoiser(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), dim=16, channels=16):
        super().__init__()
        self.text = nn.Embedding(vocab_size, dim)
        self.to_channels = nn.Linear(dim, 8)
        self.net = nn.Sequential(
            nn.Conv2d(4 + 8, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x, t, token_ids):
        pooled = self.text(token_ids).mean(dim=1)
        cond = self.to_channels(pooled).view(x.shape[0], 8, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        t_img = (t.float() / (T - 1)).view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_img, cond], dim=1))


model = TextConditionedDenoiser().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
for step in range(10):
    x0, caps = make_shape_batch(8)
    x0 = x0.to(device)
    tok = tokenize(caps)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred = model(xt, t, tok)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0:
        print(step, round(loss.item(), 4))
"""),
        code(r"""
prompts = ["red circle", "blue circle", "green square", "yellow triangle"]
tok = tokenize(prompts)

@torch.no_grad()
def cond_call(x, t, cond):
    return model(x, t, cond)

samples = ddpm_sample(model, (len(prompts), 3, 32, 32), cond=tok)
show_images(samples.cpu(), prompts, n=len(prompts))
"""),
        md(r"""
## Interview Takeaway

This is pooled text conditioning. It is enough for simple labels but not enough for long prompts, spatial relations, or text rendering. Qwen-like systems need token-level multimodal conditioning.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw6_text_conditioning/hw6_text_conditioning.ipynb", cells)


def hw7():
    cells = [
        md(r"""
# HW7 - Classifier-Free Guidance

Question: how can prompts become stronger during sampling?

""" + GPU_NOTE + r"""

Sources:

- Classifier-Free Diffusion Guidance: https://arxiv.org/abs/2207.12598
- Latent Diffusion: https://arxiv.org/abs/2112.10752
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(TEXT_UTILS),
        code(DIFFUSION),
        code(r"""
class CFGDenoiser(nn.Module):
    def __init__(self, vocab_size=len(VOCAB), dim=16, channels=16):
        super().__init__()
        self.text = nn.Embedding(vocab_size, dim)
        self.to_channels = nn.Linear(dim, 8)
        self.net = nn.Sequential(
            nn.Conv2d(4 + 8, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x, t, token_ids):
        if token_ids is None:
            token_ids = torch.full((x.shape[0], 2), stoi["<null>"], dtype=torch.long, device=x.device)
        pooled = self.text(token_ids).mean(dim=1)
        cond = self.to_channels(pooled).view(x.shape[0], 8, 1, 1).expand(-1, -1, x.shape[2], x.shape[3])
        t_img = (t.float() / (T - 1)).view(-1, 1, 1, 1).expand(-1, 1, x.shape[2], x.shape[3])
        return self.net(torch.cat([x, t_img, cond], dim=1))


model = CFGDenoiser().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
for step in range(10):
    x0, caps = make_shape_batch(8)
    x0 = x0.to(device)
    tok = tokenize(caps, drop_prob=0.15)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred = model(xt, t, tok)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0:
        print(step, round(loss.item(), 4))
"""),
        code(r"""
prompt = ["red circle"] * 2
tok = tokenize(prompt)

@torch.no_grad()
def cond_fn(x, t, cond):
    return model(x, t, cond)

all_samples, titles = [], []
for scale in [0.0, 3.0]:
    samples = ddpm_sample(
        model,
        (2, 3, 32, 32),
        cond=tok,
        guidance_scale=scale,
        cond_fn=cond_fn,
    )
    all_samples.append(samples.cpu())
    titles += [f"scale {scale}"] * 2

show_images(torch.cat(all_samples), titles, n=4, size=1.8)
"""),
        md(r"""
## Interview Takeaway

Classifier-free guidance runs conditional and unconditional predictions, then pushes the sample toward the conditional direction. Too little guidance ignores the prompt; too much guidance can reduce diversity or create artifacts.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw7_cfg/hw7_cfg.ipynb", cells)


def hw8():
    cells = [
        md(r"""
# HW8 - Tiny DiT

Question: how does a transformer replace a U-Net?

""" + GPU_NOTE + r"""

Sources:

- DiT: https://arxiv.org/abs/2212.09748
- Official DiT repo: https://github.com/facebookresearch/DiT
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(TEXT_UTILS),
        code(DIFFUSION),
        code(r"""
PATCH = 4


def patchify(x, patch=PATCH):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens, patch=PATCH, channels=3, size=32):
    b, n, d = tokens.shape
    side = size // patch
    x = tokens.reshape(b, side, side, patch, patch, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, size, size)


x, caps = make_shape_batch(2)
tokens = patchify(x)
print("image:", x.shape, "tokens:", tokens.shape, "recon:", unpatchify(tokens).shape)
"""),
        code(r"""
class TinyDiT(nn.Module):
    def __init__(self, token_dim=48, model_dim=32, depth=1, text_dim=16):
        super().__init__()
        self.in_proj = nn.Linear(token_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, token_dim)
        self.pos = nn.Parameter(torch.randn(1, 64, model_dim) * 0.02)
        self.text = nn.Embedding(len(VOCAB), text_dim)
        self.cond = nn.Linear(text_dim + 1, model_dim)
        layer = nn.TransformerEncoderLayer(model_dim, nhead=4, dim_feedforward=64, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, t, token_ids):
        toks = patchify(x)
        h = self.in_proj(toks) + self.pos[:, :toks.shape[1]]
        pooled = self.text(token_ids).mean(dim=1)
        t_scaled = t.float()[:, None] / (T - 1)
        cond = self.cond(torch.cat([pooled, t_scaled], dim=1))[:, None, :]
        h = self.blocks(h + cond)
        return unpatchify(self.out_proj(h))


model = TinyDiT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
for step in range(5):
    x0, caps = make_shape_batch(8)
    x0 = x0.to(device)
    tok = tokenize(caps)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred = model(xt, t, tok)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0:
        print(step, round(loss.item(), 4))
"""),
        code(r"""
prompts = ["red circle", "blue square", "green triangle", "yellow circle"]
samples = ddpm_sample(model, (len(prompts), 3, 32, 32), cond=tokenize(prompts)).cpu()
show_images(samples, prompts, n=len(prompts))
"""),
        md(r"""
## Interview Takeaway

DiT replaces convolutional feature maps with patch tokens. Position embeddings tell the transformer where patches are. Conditioning is injected through timestep and text vectors.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw8_tiny_dit/hw8_tiny_dit.ipynb", cells)


def hw9():
    cells = [
        md(r"""
# HW9 - Tiny MMDiT-Style Conditioning

Question: how do text tokens and image tokens interact?

""" + GPU_NOTE + r"""

Sources:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- DiT: https://arxiv.org/abs/2212.09748
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(TEXT_UTILS),
        code(DIFFUSION),
        code(r"""
PATCH = 4


def patchify(x, patch=PATCH):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens, patch=PATCH, channels=3, size=32):
    b, n, d = tokens.shape
    side = size // patch
    x = tokens.reshape(b, side, side, patch, patch, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, size, size)
"""),
        code(r"""
class TinyMMDiT(nn.Module):
    def __init__(self, token_dim=48, model_dim=32, depth=1):
        super().__init__()
        self.img_in = nn.Linear(token_dim, model_dim)
        self.img_out = nn.Linear(model_dim, token_dim)
        self.text = nn.Embedding(len(VOCAB), model_dim)
        self.time = nn.Linear(1, model_dim)
        self.pos_img = nn.Parameter(torch.randn(1, 64, model_dim) * 0.02)
        self.pos_txt = nn.Parameter(torch.randn(1, 2, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(model_dim, nhead=4, dim_feedforward=64, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, t, token_ids):
        img = self.img_in(patchify(x)) + self.pos_img
        txt = self.text(token_ids) + self.pos_txt
        time = self.time((t.float() / (T - 1))[:, None])[:, None, :]
        tokens = torch.cat([txt + time, img + time], dim=1)
        out = self.blocks(tokens)
        img_out = out[:, txt.shape[1]:]
        return unpatchify(self.img_out(img_out))


model = TinyMMDiT().to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
"""),
        code(r"""
x0, caps = make_shape_batch(4)
tok = tokenize(caps)
print("caption tokens:", tok.shape)
print("image latent tokens:", patchify(x0).shape)

for step in range(5):
    x0, caps = make_shape_batch(8)
    x0 = x0.to(device)
    tok = tokenize(caps)
    t = torch.randint(0, T, (x0.shape[0],), device=device)
    xt, eps = q_sample(x0, t)
    pred = model(xt, t, tok)
    loss = F.mse_loss(pred, eps)
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5 == 0:
        print(step, round(loss.item(), 4))
"""),
        code(r"""
prompts = ["red circle", "blue square", "green triangle", "yellow triangle"]
samples = ddpm_sample(model, (len(prompts), 3, 32, 32), cond=tokenize(prompts)).cpu()
show_images(samples, prompts, n=len(prompts))
"""),
        md(r"""
## Interview Takeaway

Pooled text conditioning compresses the whole prompt into one vector. MMDiT-style token mixing keeps text tokens and image tokens in the same reasoning space, which is closer to Qwen-like multimodal conditioning.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw9_tiny_mmdit/hw9_tiny_mmdit.ipynb", cells)


def hw10():
    cells = [
        md(r"""
# HW10 - Final Tiny Qwen-Like Text-To-Image

Question: can the pieces become one tiny end-to-end model?

""" + GPU_NOTE + r"""

Sources:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
- DiT: https://arxiv.org/abs/2212.09748
- Latent Diffusion: https://arxiv.org/abs/2112.10752

This is not a quality model. It is a compact proof that you understand the interfaces: text tokens, image tokens, timestep, denoising objective, guidance, and sampling.
"""),
        code(COMMON_SETUP),
        code(SHAPES),
        code(TEXT_UTILS),
        code(DIFFUSION),
        code(r"""
PATCH = 4


def patchify(x, patch=PATCH):
    b, c, h, w = x.shape
    x = x.reshape(b, c, h // patch, patch, w // patch, patch)
    x = x.permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // patch) * (w // patch), patch * patch * c)


def unpatchify(tokens, patch=PATCH, channels=3, size=32):
    b, n, d = tokens.shape
    side = size // patch
    x = tokens.reshape(b, side, side, patch, patch, channels)
    x = x.permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, channels, size, size)
"""),
        code(r"""
class TinyQwenLikeT2I(nn.Module):
    def __init__(self, mode="token", token_dim=48, model_dim=32, depth=1):
        super().__init__()
        assert mode in {"none", "pooled", "token"}
        self.mode = mode
        self.img_in = nn.Linear(token_dim, model_dim)
        self.img_out = nn.Linear(model_dim, token_dim)
        self.text = nn.Embedding(len(VOCAB), model_dim)
        self.time = nn.Linear(1, model_dim)
        self.pos_img = nn.Parameter(torch.randn(1, 64, model_dim) * 0.02)
        self.pos_txt = nn.Parameter(torch.randn(1, 2, model_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(model_dim, nhead=4, dim_feedforward=64, batch_first=True, activation="gelu")
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x, t, token_ids=None):
        if token_ids is None:
            token_ids = torch.full((x.shape[0], 2), stoi["<null>"], dtype=torch.long, device=x.device)
        img = self.img_in(patchify(x)) + self.pos_img
        time = self.time((t.float() / (T - 1))[:, None])[:, None, :]
        img = img + time
        if self.mode == "none":
            tokens = img
            txt_len = 0
        elif self.mode == "pooled":
            pooled = self.text(token_ids).mean(dim=1, keepdim=True)
            tokens = torch.cat([pooled + time, img], dim=1)
            txt_len = 1
        else:
            txt = self.text(token_ids) + self.pos_txt + time
            tokens = torch.cat([txt, img], dim=1)
            txt_len = txt.shape[1]
        out = self.blocks(tokens)
        img_out = out[:, txt_len:]
        return unpatchify(self.img_out(img_out))


def train_model(mode="token", steps=900, cond_drop=0.15):
    model = TinyQwenLikeT2I(mode=mode).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    for step in range(steps):
        x0, caps = make_shape_batch(8)
        x0 = x0.to(device)
        drop = cond_drop if mode != "none" else 1.0
        tok = tokenize(caps, drop_prob=drop)
        t = torch.randint(0, T, (x0.shape[0],), device=device)
        xt, eps = q_sample(x0, t)
        pred = model(xt, t, None if mode == "none" else tok)
        loss = F.mse_loss(pred, eps)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 300 == 0:
            print(mode, step, round(loss.item(), 4))
    return model
"""),
        code(r"""
# Train the final token-conditioned model.
# For a full ablation, also train modes "none" and "pooled".
model = train_model("token", steps=5)
"""),
        code(r"""
@torch.no_grad()
def model_call(x, t, cond):
    return model(x, t, cond)


prompts = ["red circle", "blue square"]
tok = tokenize(prompts)
samples = ddpm_sample(
    model,
    (len(prompts), 3, 32, 32),
    cond=tok,
    guidance_scale=3.0,
    cond_fn=model_call,
).cpu()
show_images(samples, prompts, n=len(prompts))
"""),
        code(r"""
# Optional ablation cell. Run when you want interview evidence.
RUN_ABLATIONS = False
if RUN_ABLATIONS:
    results = {}
    for mode in ["none", "pooled", "token"]:
        m = train_model(mode, steps=5)
        cond = None if mode == "none" else tokenize(prompts)
        imgs = ddpm_sample(m, (len(prompts), 3, 32, 32), cond=cond).cpu()
        results[mode] = imgs
        show_images(imgs, [mode] * len(prompts), n=len(prompts))
"""),
        md(r"""
## What This Proves

- Text is tokenized and embedded.
- Images are patchified into tokens.
- Time enters every denoising step.
- The transformer predicts the image update.
- Sampling repeatedly converts noise into an image.

## What Real Qwen Adds

- much stronger condition encoder
- production VAE with high compression and high reconstruction quality
- larger MMDiT
- large-scale curated text/image data
- curriculum for text rendering and prompt following
- editing tasks with semantic and reconstructive image paths
"""),
    ]
    write_notebook(ROOT / "homeworks/hw10_final_tiny_qwen/hw10_final_tiny_qwen.ipynb", cells)


def hw11():
    cells = [
        md(r"""
# HW11 - Qwen Interview Mapping

Question: which parts of Qwen-Image are architecture, data, scale, or engineering?

Sources:

- Qwen-Image Technical Report: https://arxiv.org/abs/2508.02324
- Qwen-Image-VAE-2.0: https://arxiv.org/abs/2605.13565
- Qwen-Image-2.0: https://arxiv.org/abs/2605.10730
- Qwen-Image-Bench: https://arxiv.org/abs/2605.28091
"""),
        md(r"""
## Mapping Table

| Qwen topic | Tiny repo version | What to say |
| --- | --- | --- |
| Text encoder | word embeddings or tiny token embeddings | Real systems need strong language/VL features for prompt following. |
| VAE | tiny AE/VAE | VAE quality caps reconstruction, text rendering, and latent diffusability. |
| DiT | tiny patch transformer | Image latents become patch tokens processed by transformer blocks. |
| MMDiT | text tokens + image tokens in one transformer | Token-level conditioning is closer to real multimodal generation. |
| Objective | DDPM or flow toy objective | The key is predicting a useful update in latent space across timesteps. |
| CFG | conditional/unconditional predictions | Guidance is a sampling-time tradeoff between prompt adherence and artifacts. |
| Data curriculum | synthetic shapes | Real Qwen uses curation, balancing, synthesis, and progressive text-rendering tasks. |
| Editing | not implemented in final model | Qwen editing uses both semantic and reconstructive image representations. |
"""),
        code(r"""
interview_questions = {
    "Why latent space?": "Cheaper generation, but VAE reconstruction and diffusability become bottlenecks.",
    "Why DiT/MMDiT?": "Transformers treat image patches and text tokens as sequences, making multimodal conditioning natural.",
    "Why text rendering is hard?": "It needs visual precision, language understanding, layout, OCR-like supervision, and data curriculum.",
    "Why editing needs dual encoding?": "Semantic encoders preserve meaning; VAE latents preserve visual details.",
    "Diffusion vs flow?": "Diffusion learns denoising; flow learns velocity. Both define iterative generation paths.",
    "What does the sampler do?": "It repeatedly applies model predictions to move from noise toward a data/latent sample.",
}

for q, a in interview_questions.items():
    print(f"Q: {q}\nA: {a}\n")
"""),
        md(r"""
## Interview Gaps To Be Honest About

- Toy data does not test real prompt following.
- Synthetic shapes do not cover text rendering or multilingual typography.
- The tiny VAE is not comparable to Qwen-Image-VAE-2.0.
- The tiny transformer demonstrates interfaces, not scale behavior.
- Real system quality comes from architecture plus data, curriculum, evaluation, and deployment engineering.
"""),
        md(r"""
## Strong Closing Explanation

The tiny repo proves that I understand the interfaces and training logic: text becomes tokens, images become latents/tokens, the timestep controls the corruption level, the transformer predicts an update, and sampling iteratively converts noise into an image. Qwen's hard parts are making every interface strong at scale: condition encoding, VAE reconstruction and diffusability, MMDiT capacity, data curriculum, and evaluation for prompt following, text rendering, and editing consistency.
"""),
    ]
    write_notebook(ROOT / "homeworks/hw11_qwen_mapping/hw11_qwen_mapping.ipynb", cells)


def main():
    hw0()
    hw1()
    hw2()
    hw3()
    hw4()
    hw5()
    hw6()
    hw7()
    hw8()
    hw9()
    hw10()
    hw11()


if __name__ == "__main__":
    main()
