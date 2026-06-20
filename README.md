# image-gen-theory

Karpathy-style notebooks for understanding modern image generation deeply enough to discuss Qwen-Image-like systems technically.

The repo is built around one idea: every paper concept must become a small runnable experiment before it becomes interview language.

## What This Repo Should Prove

By the end, you should be able to explain and implement a tiny version of the core Qwen-Image stack:

```text
caption
  -> tokenizer
  -> text encoder / text tokens

noise image or latent
  -> patch tokens
  -> timestep conditioning
  -> tiny DiT / MMDiT
  -> predicted noise or velocity
  -> sampler loop
  -> generated image or latent
```

The tiny model is not meant to be beautiful. It is meant to prove that you understand the interfaces, training targets, sampling behavior, failure modes, and tradeoffs.

## Run Order

Open notebooks from `homeworks/` in this order:

| HW | Notebook | Core idea | Interview outcome |
| --- | --- | --- | --- |
| 0 | [system map](homeworks/hw0_system_map/hw0_system_map.ipynb) | Full architecture shape | Explain the Qwen-like pipeline end to end. |
| 1 | [2D flow](homeworks/hw1_2d_flow/hw1_2d_flow.ipynb) | Generation as motion | Explain velocity prediction and ODE-style sampling. |
| 2 | [2D diffusion](homeworks/hw2_2d_diffusion/hw2_2d_diffusion.ipynb) | Generation as denoising | Explain forward noising, reverse denoising, and noise prediction. |
| 3 | [tiny images](homeworks/hw3_tiny_images/hw3_tiny_images.ipynb) | Images as high-dimensional points | Explain why image models need spatial or token structure. |
| 4 | [autoencoder/VAE](homeworks/hw4_autoencoder_vae/hw4_autoencoder_vae.ipynb) | Compression and reconstruction | Explain why VAE quality caps generation quality. |
| 5 | [latent generation](homeworks/hw5_latent_generation/hw5_latent_generation.ipynb) | Generate latents, then decode | Compare pixel-space and latent-space generation. |
| 6 | [text conditioning](homeworks/hw6_text_conditioning/hw6_text_conditioning.ipynb) | Text controls generation | Explain tokenizer, embeddings, and pooled conditioning. |
| 7 | [CFG](homeworks/hw7_cfg/hw7_cfg.ipynb) | Prompt strength at sampling time | Explain conditional/unconditional prediction mixing. |
| 8 | [tiny DiT](homeworks/hw8_tiny_dit/hw8_tiny_dit.ipynb) | Images as patch tokens | Explain patchify, position embeddings, and transformer denoising. |
| 9 | [tiny MMDiT](homeworks/hw9_tiny_mmdit/hw9_tiny_mmdit.ipynb) | Text and image tokens interact | Compare pooled conditioning with token-level conditioning. |
| 10 | [tiny Qwen-like T2I](homeworks/hw10_final_tiny_qwen/hw10_final_tiny_qwen.ipynb) | End-to-end toy system | Show text + noise -> image through a tiny transformer sampler. |
| 11 | [Qwen mapping](homeworks/hw11_qwen_mapping/hw11_qwen_mapping.ipynb) | Paper-to-code mapping | Separate architecture, data, scale, evaluation, and engineering. |

## Interview Map

### Diffusion vs Flow Matching

| Question | Diffusion | Flow matching |
| --- | --- | --- |
| What is learned? | Denoising/noise prediction over timesteps. | Velocity field from source distribution to data. |
| Sampling picture | Start from noise and reverse the noising chain. | Start from noise and integrate the learned velocity. |
| Debug visual | Noising and denoising snapshots. | Sample trajectories and vector fields. |
| Key weakness | Many steps; schedule matters. | Path and solver choices matter. |
| Qwen relevance | Many modern image systems still use diffusion language and timestep-conditioned denoisers. | Flow-style objectives are important for modern generative training and sampling intuition. |

### Pixel vs Latent Generation

| Pixel-space | Latent-space |
| --- | --- |
| Easier to understand. | Closer to modern high-resolution systems. |
| Expensive as resolution grows. | Cheaper because the generator models compressed representations. |
| No decoder bottleneck. | VAE reconstruction and latent diffusability become critical. |
| Good for first toy experiments. | Necessary for Qwen-like reasoning. |

### U-Net vs DiT vs MMDiT

| Backbone | Main idea | What to understand |
| --- | --- | --- |
| U-Net | Convolutional denoiser with spatial inductive bias. | Strong for local image structure; old latent diffusion style. |
| DiT | Transformer over image or latent patches. | Images become token sequences; timestep and conditioning enter token processing. |
| MMDiT | Multimodal transformer over text/image streams. | Text tokens and image tokens interact more directly than pooled conditioning. |

### Pooled Text vs Token-Level Conditioning

| Pooled text | Token-level text |
| --- | --- |
| Simple and enough for labels like "red circle". | Needed for long prompts, spatial relations, multilingual text, and layout. |
| Compresses all words into one vector. | Preserves sequence structure and token-level detail. |
| Good for HW6. | Closer to Qwen-Image-style MMDiT. |

## Qwen-Image Technical Topics To Know

Grounded in the public reports:

- Qwen-Image targets complex text rendering and precise image editing, not only generic image aesthetics.
- The technical report emphasizes data collection, filtering, annotation, synthesis, balancing, and progressive training for text rendering.
- Qwen-Image editing uses T2I, TI2I, and I2I reconstruction tasks to align representations.
- For editing, the original image is fed through both a semantic path and a reconstructive VAE path.
- Qwen-Image-VAE-2.0 emphasizes reconstruction fidelity and diffusability, meaning the latent must both reconstruct well and be easy for the diffusion/DiT model to learn.
- Qwen-Image-2.0 raises the bar further with stronger condition encoding, MMDiT-style joint condition-target modeling, long prompts, text-rich content, multilingual typography, and deployment concerns.

Interview framing:

```text
Toy repo: proves I understand the interfaces and objectives.
Real Qwen: wins by scaling every interface with better VAE, condition encoder, MMDiT, data, curriculum, evaluation, and engineering.
```

## Resource Map

| Resource | Topic | Use |
| --- | --- | --- |
| [Qwen-Image Technical Report](https://arxiv.org/abs/2508.02324) | Qwen architecture, text rendering, editing | Main target paper. |
| [Qwen-Image-VAE-2.0](https://arxiv.org/abs/2605.13565) | VAE reconstruction and diffusability | Understand the latent bottleneck. |
| [Qwen-Image-2.0](https://arxiv.org/abs/2605.10730) | Updated system-level Qwen direction | Understand newer challenges and scaling. |
| [Qwen-Image-Bench](https://arxiv.org/abs/2605.28091) | Evaluation | Understand creator-centric evaluation. |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | Flow matching notebooks | First flow-matching reference. |
| [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | Flow matching implementation | Modern library and examples. |
| [DDPM](https://arxiv.org/abs/2006.11239) | Diffusion | Original denoising diffusion objective. |
| [Flow Matching](https://arxiv.org/abs/2210.02747) | Flow objective | Velocity-field formulation. |
| [Latent Diffusion](https://arxiv.org/abs/2112.10752) | Latent image generation | Why generate compressed latents. |
| [DiT](https://arxiv.org/abs/2212.09748) | Transformer denoising | Why patch-token transformers replaced many U-Nets. |
| [Diffusion Explainer](https://poloclub.github.io/diffusion-explainer/) | Visual intuition | Use before heavy equations. |

## Notebook Standard

Each notebook should answer:

- What problem is this approach solving?
- What does the model predict?
- What is the training target?
- What does the sampler do?
- What tensor shapes move through the model?
- What fails first?
- What does this teach about Qwen-like systems?

Every notebook should include:

- source links
- dataset visualization
- tensor shape checks
- training loop
- sampling loop
- failure or limitation notes
- interview takeaway

## Setup

The checked-in notebook outputs are CPU-smoke outputs. They verify that code executes, figures render, tensors have the expected shapes, and the sampler path is wired correctly. Many generated image grids are intentionally undertrained and may look like noise; run the same notebooks longer on a GPU for meaningful samples.

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Regenerate notebooks:

```bash
python3 scripts/create_homework_notebooks.py
```

Execute every notebook and save outputs in place:

```bash
python3 scripts/execute_notebooks.py
```

Validate repo health:

```bash
python3 scripts/check_repo.py
```

Review executed notebook outputs:

```bash
python3 scripts/review_notebook_outputs.py
```

Current environment note: this repo can validate notebook JSON and Python syntax without PyTorch installed, but executing the notebooks requires `torch`, `matplotlib`, and Jupyter.
