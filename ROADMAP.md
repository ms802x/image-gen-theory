# Karpathy-Style Roadmap To Tiny Qwen-Like Text-To-Image

End goal: build a tiny Qwen-Image-like text-to-image system for understanding, not quality.

The target is not image editing yet. The target is a minimal text-to-image architecture with the same broad ingredients:

- a latent image representation
- a text condition encoder
- a diffusion or flow-matching generative core
- a transformer/DiT-style denoiser or velocity model
- a small training dataset with text-image pairs
- sampling that turns text + noise into an image

## Rules

- Code first, theory second.
- Each theory idea must connect to a runnable experiment.
- Start in 2D or tiny images before real image datasets.
- Prefer grounded public repos, notebooks, papers, and demos over invented notebooks.
- Keep experiments small enough to run on CPU when possible.
- When GPU becomes useful, use it only after the CPU version teaches the idea.

## Phase 0: Orientation

Goal: know the system shape before touching papers.

Build/read:

- Draw the pipeline by hand:
  - text prompt
  - text encoder
  - noise latent
  - timestep
  - transformer denoiser/velocity model
  - VAE decoder or direct pixel decoder
  - image

Read:

- Qwen-Image Technical Report abstract and architecture sections.
- Do not read training-scale details yet.

Checkpoint:

- You can explain the model in one sentence:
  "A text encoder conditions a latent diffusion/flow transformer that maps noise into image latents, then a decoder turns latents into pixels."

## Phase 1: 2D Flow Matching

Goal: understand generation as moving noise into data.

Run/inspect:

- TorchCFM 2D tutorials.
- Start with moons, circles, Gaussian blobs, or spirals.

Implement only after running a grounded version:

- Tiny MLP velocity model.
- Input: `x_t`, `t`.
- Output: velocity.
- Sample with simple Euler steps.

Theory to read:

- Flow Matching for Generative Modeling:
  - intro
  - conditional flow matching objective
  - ODE sampling picture

Checkpoint:

- You can animate random points becoming the target 2D distribution.
- You understand that the network predicts "which way should this point move now?"

## Phase 2: 2D Diffusion

Goal: understand generation as denoising.

Run/inspect:

- A small DDPM implementation.
- Use public tutorials as reference, but keep the experiment tiny.

Implement:

- Add noise to 2D points at timestep `t`.
- Tiny MLP predicts noise or clean point.
- Sampling starts from noise and denoises step by step.

Theory to read:

- DDPM paper:
  - forward noising process
  - reverse denoising process
  - noise prediction loss

Checkpoint:

- You can explain the difference:
  - diffusion predicts how to remove noise
  - flow matching predicts motion from noise to data

## Phase 3: Tiny Images Without Text

Goal: move from points to images without adding text complexity yet.

Dataset options:

- MNIST
- Fashion-MNIST
- synthetic 16x16 shapes

Implement:

- Pixel-space DDPM or flow model first.
- Tiny U-Net or tiny transformer.
- Generate unconditional images.

Theory to read:

- Diffusion Explainer for intuition.
- DDPM details only where they match your code.

Checkpoint:

- You can sample recognizable tiny images from noise.
- You understand image tensors as just high-dimensional points.

## Phase 4: Latent Space And VAE

Goal: understand why Qwen-like systems generate latents, not pixels.

Build/read:

- Train a tiny autoencoder or VAE on MNIST/synthetic shapes.
- Encode image to latent.
- Decode latent back to image.
- Then train diffusion/flow in latent space.

Theory to read:

- Latent Diffusion paper:
  - why latent diffusion is cheaper
  - VAE encoder/decoder role
  - conditioning overview

Checkpoint:

- You can generate a latent from noise and decode it into an image.
- You understand that VAE quality limits final image quality.

## Phase 5: Text Conditioning

Goal: make text affect generation.

Dataset options:

- Synthetic shapes with captions:
  - "red circle"
  - "blue square"
  - "three green dots"
- MNIST with labels converted to text:
  - "digit zero"
  - "digit seven"

Implement:

- Tiny tokenizer.
- Tiny text encoder:
  - embedding average first
  - then small transformer later
- Condition the diffusion/flow model with text embedding.

Theory to read:

- Classifier-free guidance explanation.
- Cross-attention basics if using cross-attention.

Checkpoint:

- Prompt changes the generated image.
- You can overfit a tiny dataset intentionally.

## Phase 6: Tiny DiT

Goal: replace U-Net/MLP with a transformer-style image generator.

Implement:

- Patchify tiny image latents into tokens.
- Add timestep embedding.
- Add text conditioning.
- Use transformer blocks to predict noise or velocity.
- Unpatchify back to latent/image shape.

Theory to read:

- DiT paper:
  - patch tokens
  - timestep conditioning
  - transformer denoiser
  - scaling intuition

Checkpoint:

- You have a tiny text-conditioned DiT that can generate simple images.
- You understand why modern systems moved from U-Nets toward transformers.

## Phase 7: Tiny MMDiT / Qwen-Like Conditioning

Goal: mimic the broad Qwen-Image architecture shape.

Implement:

- Separate text tokens and image latent tokens.
- A small multimodal transformer where text/image tokens interact.
- Predict velocity or noise for image latent tokens only.

Study:

- Qwen-Image Technical Report:
  - MMDiT
  - text rendering motivation
  - conditioning pipeline
  - multi-stage training
- Qwen-Image-2.0 report only after the small version works.

Checkpoint:

- You can point to your tiny model and say:
  - this is the text encoder
  - this is the latent representation
  - this is the multimodal diffusion transformer
  - this is the sampler

## Phase 8: Scaling Lessons

Goal: understand what the paper adds beyond the toy version.

Study:

- Qwen-Image data pipeline.
- Qwen-Image training curriculum.
- Qwen-Image VAE report.
- Prompt following and text rendering benchmarks.

Do not attempt first:

- billion-scale data
- full OCR-heavy text rendering
- production VAE training
- full Qwen-quality model

Checkpoint:

- You understand which parts are science/architecture and which parts are scale/data/engineering.

## Final Tiny Architecture

Minimum successful project:

```text
caption -> tiny text encoder -> text tokens
noise latent -> patch tokens
timestep -> time embedding
text tokens + image tokens -> tiny MMDiT/DiT
predicted velocity/noise -> sampler loop
generated latent -> decoder -> image
```

Recommended first dataset:

```text
16x16 or 32x32 synthetic colored shapes with generated captions
```

Why synthetic first:

- perfect labels
- fast CPU training
- easy debugging
- you can know immediately whether prompt following works

## Reading Priority

1. TorchCFM 2D examples.
2. Flow Matching for Generative Modeling.
3. DDPM.
4. Latent Diffusion.
5. DiT.
6. Qwen-Image Technical Report.
7. Qwen-Image-VAE-2.0.
8. Qwen-Image-2.0.

Only read math deeply after you can point to the exact line of code it explains.
