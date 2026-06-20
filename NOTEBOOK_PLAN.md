# Notebook Plan

This repo should become a structured set of Karpathy-style notebooks for understanding modern image generation, ending in a tiny Qwen-like text-to-image model.

The notebooks are not meant to be polished demos. They are lab notebooks: small, inspectable, heavily instrumented, and designed to build intuition.

## Notebook Creation Rule

Before creating a notebook:

1. Search for public notebooks, repos, demos, and paper code.
2. Run or inspect the strongest existing examples.
3. Compare at least two sources when possible.
4. Write a short source note:
   - what was reused
   - what was simplified
   - what was changed
   - what remains unclear

Create a new notebook only when:

- no good public notebook exists for the exact learning step
- multiple notebooks exist but none connect the ideas clearly
- a small bridge experiment is needed between two concepts
- the homework requires instrumentation that upstream notebooks do not provide

## Notebook Style

Each notebook should have this shape:

```text
0. Goal
1. Sources
2. Minimal setup
3. Dataset visualization
4. Forward/noising/interpolation process
5. Model definition
6. Training loop
7. Sampling loop
8. Visual diagnostics
9. What changed when parameters changed
10. What this teaches
11. What this does not teach yet
```

Every notebook should include:

- tensor shape printouts
- plots before training
- plots during or after training
- at least one intentional overfit test
- at least one failure case
- comments connecting code to theory
- a final comparison table when there are multiple approaches

Avoid:

- big unexplained abstractions
- hidden training magic
- long math derivations before code
- copying a full production repo into a notebook without reducing it
- treating loss curves as proof that the concept is understood

## Homework Notebook Sequence

| Homework | Notebook target | Main intuition | Main comparison |
| --- | --- | --- | --- |
| HW0 | `hw0_system_map.ipynb` or notes | Full text-to-image system shape | Qwen-scale system vs tiny system |
| HW1 | `hw1_2d_flow.ipynb` | Generation as motion | flow path choices |
| HW2 | `hw2_2d_diffusion.ipynb` | Generation as denoising | diffusion vs flow |
| HW3 | `hw3_tiny_images.ipynb` | Images as high-dimensional points | MLP vs U-Net vs tiny transformer |
| HW4 | `hw4_autoencoder_vae.ipynb` | Compression and reconstruction | autoencoder vs VAE |
| HW5 | `hw5_latent_generation.ipynb` | Generate latents, then decode | pixel generation vs latent generation |
| HW6 | `hw6_text_conditioning.ipynb` | Text as conditioning signal | label conditioning vs text conditioning |
| HW7 | `hw7_cfg.ipynb` | Prompt strength at sampling time | low vs high guidance |
| HW8 | `hw8_tiny_dit.ipynb` | Image latents as tokens | U-Net-style vs DiT-style denoiser |
| HW9 | `hw9_tiny_mmdit.ipynb` | Text tokens and image tokens interacting | pooled text embedding vs token-level conditioning |
| HW10 | `hw10_tiny_qwen_t2i.ipynb` plus scripts | End-to-end tiny Qwen-like T2I | toy architecture vs Qwen paper |
| HW11 | `qwen_mapping.md` | Paper understanding | architecture vs data vs scale vs engineering |

## Source Strategy By Topic

### Flow matching

Primary sources:

- TorchCFM
- Flow Matching for Generative Modeling
- facebookresearch/flow_matching

Notebook goal:

- make velocity fields visible
- compare straight-line interpolation, conditional flow matching, and sampling step count
- show why ODE sampling feels different from DDPM denoising

### Diffusion

Primary sources:

- DDPM paper
- Diffusion Explainer
- small DDPM repos or annotated tutorials

Notebook goal:

- visualize noising schedules
- train noise prediction on 2D data
- show reverse denoising step by step
- compare noise prediction vs clean-data prediction

### Latent generation

Primary sources:

- Latent Diffusion paper
- VAE references
- small autoencoder/VAE notebooks if found

Notebook goal:

- show reconstruction bottlenecks
- train generation in latent space
- explain why bad VAE reconstruction limits final image quality

### Text conditioning

Primary sources:

- classifier-free guidance references
- simple conditional diffusion examples
- synthetic captioned dataset examples if found

Notebook goal:

- show prompt embedding effects
- test prompt swapping
- test missing words and contradictory prompts
- compare class labels, pooled text embeddings, and token sequences

### DiT and MMDiT

Primary sources:

- DiT paper and implementations
- Qwen-Image Technical Report
- Qwen-Image model/repo materials

Notebook goal:

- trace patch tokens through the model
- compare image-only transformer vs text-conditioned transformer
- show the difference between pooled conditioning and token-level multimodal conditioning
- map every tiny component to the corresponding Qwen-like idea

## Required Comparison Notes

Every major notebook should answer these questions:

- What problem is this approach solving?
- What does the model predict?
- What is the training target?
- What is the sampler doing?
- What is expensive?
- What breaks first?
- What does this teach about Qwen-like systems?
- What would need scale, better data, or engineering to work seriously?

## Qwen-Specific Understanding Targets

By the final homework, the repo should make these challenges concrete:

- prompt following
- text rendering
- VAE reconstruction quality
- latent resolution and compression
- text-image alignment
- training data quality
- sampling speed vs quality
- diffusion objective vs flow objective
- U-Net vs DiT vs MMDiT tradeoffs
- pooled conditioning vs token-level conditioning
- tiny toy success vs real-world model scale

## Definition Of A Good Notebook

A good notebook should let the reader:

- run it from top to bottom
- see the data before modeling
- understand the target the model is trained on
- inspect model inputs and outputs
- watch sampling happen
- change one parameter and predict the effect
- explain the idea without quoting the paper

If a notebook does not build intuition, it is not finished.
