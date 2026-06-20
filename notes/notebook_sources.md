# Notebook Source Survey

This file tracks public notebooks, repos, demos, and papers to inspect before creating each homework notebook.

Rule: use existing notebooks first. If the repo creates a new notebook, it should cite the sources below and explain what was reused, simplified, or changed.

## HW0: System Map

Goal:

- Understand the full text-to-image system shape before implementing pieces.
- Map a Qwen-like system into a tiny version.

Primary sources:

| Source | Link | Use |
| --- | --- | --- |
| Qwen-Image Technical Report | https://arxiv.org/abs/2508.02324 | Architecture target: Qwen-style text-to-image and editing system, MMDiT, VAE, text/image conditioning, training curriculum. |
| Qwen-Image-VAE-2.0 Technical Report | https://arxiv.org/abs/2605.13565 | Why VAE reconstruction quality and latent diffusability matter. |
| DiT paper | https://arxiv.org/abs/2212.09748 | Transformer denoiser over latent/image patches. |
| Latent Diffusion paper | https://arxiv.org/abs/2112.10752 | Why text-to-image systems generate in latent space. |

What to extract:

- Component diagram:
  - tokenizer
  - text encoder
  - image encoder/VAE
  - latent/noise representation
  - timestep embedding
  - DiT/MMDiT backbone
  - sampler
  - decoder
- What the tiny project keeps:
  - text conditioning
  - latent or compressed image representation
  - transformer denoiser/velocity model
  - sampler loop
- What the tiny project drops:
  - huge dataset
  - OCR-scale text rendering
  - production VAE training
  - billion-parameter model scale

HW0 notebook/note decision:

- A markdown note is enough for HW0.
- No custom notebook needed unless a visual architecture diagram is created programmatically.

## HW1: 2D Flow Matching

Goal:

- Learn generation as motion from a simple source distribution to a data distribution.
- Make the velocity field and sampler visible.

Primary sources:

| Source | Link | Use |
| --- | --- | --- |
| TorchCFM 2D tutorials | https://github.com/atong01/conditional-flow-matching/tree/main/examples/2D_tutorials | Best source for 2D flow matching notebooks, including tutorial and model comparison notebooks. |
| TorchCFM `Flow_matching_tutorial.ipynb` | https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/Flow_matching_tutorial.ipynb | First notebook to inspect/run for conditional flow matching basics. |
| TorchCFM `tutorial_training_8_gaussians_to_moons.ipynb` | https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/tutorial_training_8_gaussians_to_moons.ipynb | Good toy transition from simple source to structured target. |
| TorchCFM `model-comparison-plotting.ipynb` | https://github.com/atong01/conditional-flow-matching/blob/main/examples/2D_tutorials/model-comparison-plotting.ipynb | Use for comparing flow matching variants visually. |
| Meta `flow_matching` examples | https://github.com/facebookresearch/flow_matching/tree/main/examples | Strong modern reference with standalone and library-based notebooks. |
| Meta `standalone_flow_matching.ipynb` | https://github.com/facebookresearch/flow_matching/blob/main/examples/standalone_flow_matching.ipynb | Concise pure PyTorch baseline to compare against TorchCFM. |
| Meta `2d_flow_matching.ipynb` | https://github.com/facebookresearch/flow_matching/blob/main/examples/2d_flow_matching.ipynb | Library-based 2D checkerboard flow matching example. |
| Flow Matching for Generative Modeling | https://arxiv.org/abs/2210.02747 | Theory source for conditional flow matching and ODE sampling. |

Comparison questions:

- What does each notebook choose as source and target distribution?
- Does it use a library abstraction or pure PyTorch?
- What path is used between noise and data?
- What exactly does the network predict?
- How many sampler steps are needed before samples look good?
- What breaks when the model is too small or sampler steps are too few?

Likely repo notebook:

- Create `homeworks/hw1_2d_flow/hw1_2d_flow.ipynb` only after inspecting TorchCFM and Meta examples.
- The notebook should be smaller than the upstream notebooks and focused on visualization:
  - source distribution
  - target distribution
  - interpolation path
  - learned velocity arrows
  - sample trajectories

## HW2: 2D Diffusion

Goal:

- Learn generation as iterative denoising.
- Compare denoising diffusion directly against HW1 flow matching.

Primary sources:

| Source | Link | Use |
| --- | --- | --- |
| DDPM paper | https://arxiv.org/abs/2006.11239 | Original denoising diffusion objective and sampling setup. |
| Diffusion Explainer | https://poloclub.github.io/diffusion-explainer/ | Visual intuition before reading equations. |
| Diffusion Explainer repo | https://github.com/poloclub/diffusion-explainer | Source/demo reference for visual explanation style. |
| Hugging Face Annotated Diffusion | https://github.com/huggingface/blog/blob/main/annotated-diffusion.md | Line-by-line DDPM walkthrough to read and cross-check. |
| lucidrains denoising-diffusion-pytorch | https://github.com/lucidrains/denoising-diffusion-pytorch | Practical PyTorch implementation reference. |
| score_sde_pytorch | https://github.com/yang-song/score_sde_pytorch | Advanced reference for score/SDE framing after DDPM basics. |

Source gap:

- There are many image DDPM notebooks, but fewer clean 2D point-cloud DDPM notebooks.
- If no high-quality 2D point notebook is found, this repo should create one.
- The custom notebook must cite DDPM, Diffusion Explainer, and at least one PyTorch implementation.

Comparison questions:

- Does the model predict noise, clean data, or score?
- What does the noising schedule look like visually?
- How does the sampler differ from HW1 Euler flow sampling?
- What happens with 10, 50, 100, and 1000 sampling steps?
- Which is easier to debug in 2D: diffusion or flow matching?

Likely repo notebook:

- Create `homeworks/hw2_2d_diffusion/hw2_2d_diffusion.ipynb`.
- It should mirror HW1 as much as possible:
  - same dataset
  - similar MLP size
  - similar visualization style
  - direct comparison table

## HW3-HW5: Tiny Images, VAE, Latent Generation

Goal:

- Move from 2D points to images.
- Understand why modern systems generate latents.

Primary sources:

| Source | Link | Use |
| --- | --- | --- |
| Latent Diffusion paper | https://arxiv.org/abs/2112.10752 | Latent-space diffusion and conditioning. |
| CompVis latent-diffusion | https://github.com/CompVis/latent-diffusion | Original LDM implementation reference. |
| lucidrains denoising-diffusion-pytorch | https://github.com/lucidrains/denoising-diffusion-pytorch | Practical PyTorch DDPM reference. |
| score_sde_pytorch | https://github.com/yang-song/score_sde_pytorch | Advanced score/SDE implementation reference. |
| Qwen-Image-VAE-2.0 Technical Report | https://arxiv.org/abs/2605.13565 | Later-stage understanding of VAE reconstruction, compression, and diffusability. |

Comparison questions:

- Pixel-space generation vs latent-space generation: what gets cheaper, and what gets worse?
- Autoencoder vs VAE: what changes in the latent distribution?
- What reconstruction errors later become generation errors?

Notebook decision:

- Likely create small custom notebooks because the goal is not production LDM but a tiny bridge:
  - train tiny autoencoder/VAE
  - freeze it
  - train tiny latent diffusion/flow
  - decode generated latents

## HW6-HW7: Text Conditioning And CFG

Goal:

- Understand how text changes generation.
- Understand why classifier-free guidance is a sampling-time control.

Primary sources to collect next:

| Source | Link | Use |
| --- | --- | --- |
| Classifier-Free Diffusion Guidance | https://arxiv.org/abs/2207.12598 | CFG objective and sampling rule. |
| Latent Diffusion paper | https://arxiv.org/abs/2112.10752 | Text conditioning and cross-attention in latent diffusion. |
| Diffusion Explainer | https://poloclub.github.io/diffusion-explainer/ | Visual intuition for prompt-guided generation. |

Comparison questions:

- Label conditioning vs text conditioning.
- Pooled text embedding vs token sequence conditioning.
- No guidance vs weak guidance vs over-guidance.

## HW8-HW10: Tiny DiT, Tiny MMDiT, Final Tiny Qwen-Like T2I

Goal:

- Move from U-Net/MLP thinking to transformer-token thinking.
- Build a tiny text-to-image model with Qwen-like ingredients.

Primary sources:

| Source | Link | Use |
| --- | --- | --- |
| DiT paper | https://arxiv.org/abs/2212.09748 | Transformer denoiser over latent patches. |
| Official DiT repo | https://github.com/facebookresearch/DiT | Implementation reference for patchify, conditioning, and transformer blocks. |
| Qwen-Image Technical Report | https://arxiv.org/abs/2508.02324 | Architecture target and Qwen-specific challenges. |
| Qwen-Image-VAE-2.0 Technical Report | https://arxiv.org/abs/2605.13565 | VAE reconstruction/diffusability challenges. |
| Meta `flow_matching` image examples | https://github.com/facebookresearch/flow_matching/tree/main/examples/image | Image flow matching reference. |

Comparison questions:

- U-Net vs DiT: where is spatial structure encoded?
- Pooled text conditioning vs token-level conditioning.
- Diffusion objective vs flow-matching objective.
- Tiny success vs real Qwen difficulty:
  - text rendering
  - multilingual prompts
  - long prompts
  - data quality
  - VAE compression
  - latent alignment
  - scaling laws

## Immediate Next Actions

1. Create `notes/hw0_system_map.md`.
2. Inspect TorchCFM HW1 notebooks locally or as submodules.
3. Decide whether to vendor upstream notebooks as submodules, copy selected notebooks with attribution, or create local bridge notebooks.
4. Build HW1 only after writing a short source note comparing TorchCFM and Meta `flow_matching`.
