# Homework Roadmap: Tiny Qwen-Like Text-To-Image

End goal: build a tiny Qwen-Image-like text-to-image model for understanding, not quality.

This is a homework sequence. Each homework should leave behind code, plots, notes, and a short conclusion. The final homework combines the earlier pieces into a much smaller Qwen-like architecture.

Target final system:

```text
caption -> tiny text encoder -> text tokens
noise latent -> patch tokens
timestep -> time embedding
text tokens + image tokens -> tiny DiT/MMDiT
predicted velocity/noise -> sampler loop
generated latent -> decoder -> image
```

Use small datasets first:

- 2D points
- MNIST
- synthetic 16x16 or 32x32 colored shapes with captions

## Course Rules

- Code first, theory second.
- Every homework must be runnable.
- Every theory reading must answer a question raised by code.
- Prefer grounded public repos, notebooks, papers, and demos over invented notebooks.
- For each homework, search for existing notebooks first. If none are good enough, create a small notebook that cites and combines multiple sources.
- Every notebook must explain differences between approaches and failure modes.
- Start CPU-friendly. Use GPU only when the CPU experiment already taught the concept.
- Overfitting tiny data is allowed and useful.
- Quality is not the goal. Understanding the moving parts is the goal.

## Homework 0: Map The System

Question:

What pieces does a Qwen-like text-to-image system need?

Task:

- Draw the full pipeline in `notes/hw0_system_map.md`.
- Identify each component:
  - text tokenizer
  - text encoder
  - image representation
  - noise process
  - timestep embedding
  - generative model
  - sampler
  - decoder

Reading:

- Qwen-Image Technical Report abstract and architecture sections.
- Do not read data scaling details yet.

Deliverable:

- A one-page diagram or bullet map.
- A short explanation of the final tiny system in your own words.
- A list of existing notebooks/repos that seem useful for the next three homeworks.

Pass condition:

- You can say what each model component consumes and produces.

## Homework 1: 2D Flow Matching

Question:

How can a model learn to move noise into data?

Task:

- Run or inspect TorchCFM's 2D examples.
- Build or adapt a tiny 2D flow-matching experiment.
- Dataset: moons, spirals, circles, or Gaussian blobs.
- Model: small MLP.
- Input: `x_t`, `t`.
- Output: velocity.
- Sampler: Euler integration.

Reading:

- TorchCFM 2D tutorial.
- Flow Matching for Generative Modeling:
  - intro
  - conditional flow matching objective
  - ODE sampling idea

Deliverable:

- Plot of source noise.
- Plot of target data.
- Plot or animation of samples moving from noise to data.
- Notes explaining what the velocity model predicts.
- Source note comparing the notebooks/repos inspected before writing code.

Pass condition:

- Random 2D noise visibly becomes the target distribution.

## Homework 2: 2D Diffusion

Question:

How is denoising different from flow?

Task:

- Build a tiny 2D DDPM-style experiment.
- Dataset: same as Homework 1.
- Add noise to clean points at random timestep `t`.
- Train an MLP to predict noise or clean point.
- Sample by iterative denoising.

Reading:

- DDPM paper:
  - forward noising
  - reverse denoising
  - noise prediction loss

Deliverable:

- Plot of data at several noise levels.
- Plot or animation of reverse denoising.
- Short comparison: diffusion vs flow matching.
- Source note explaining which DDPM notebooks or repos were inspected.

Pass condition:

- You can explain:
  - diffusion predicts how to remove noise
  - flow matching predicts how a sample should move

## Homework 3: Tiny Unconditional Images

Question:

What changes when the data is an image instead of a 2D point?

Task:

- Train a tiny unconditional image generator.
- Dataset options:
  - MNIST
  - Fashion-MNIST
  - synthetic 16x16 shapes
- Model options:
  - tiny U-Net
  - tiny image transformer
  - tiny MLP for very small images
- Use diffusion or flow matching.

Reading:

- Diffusion Explainer.
- DDPM sections that match your code.

Deliverable:

- Grid of generated samples.
- Training loss curve.
- Notes explaining image tensors as high-dimensional points.
- Comparison note: what changed from 2D points to images.

Pass condition:

- The model generates recognizable tiny images or clearly learns the synthetic distribution.

## Homework 4: Tiny Autoencoder / VAE

Question:

Why do modern text-to-image models generate latents instead of pixels?

Task:

- Train a tiny autoencoder or VAE.
- Dataset: same as Homework 3.
- Encode images into latents.
- Decode latents back to images.
- Measure reconstruction quality by visual comparison.

Reading:

- Latent Diffusion paper:
  - VAE role
  - latent-space generation
  - why it is cheaper than pixel generation

Deliverable:

- Original vs reconstructed image grid.
- Latent size documentation.
- Notes on what information the latent preserves or loses.
- Failure note: examples where reconstruction loses information.

Pass condition:

- Reconstructions are good enough that a latent generator could use them.

## Homework 5: Latent Diffusion Or Latent Flow

Question:

Can we generate images by generating latents?

Task:

- Freeze the autoencoder/VAE from Homework 4.
- Train a diffusion or flow model in latent space.
- Decode generated latents back to images.

Reading:

- Latent Diffusion paper sections on latent generative modeling.
- Revisit Flow Matching or DDPM depending on which objective you use.

Deliverable:

- Generated latent samples decoded into images.
- Comparison with pixel-space generation from Homework 3.
- Notes on speed, quality, and failure modes.
- Explanation of what latent generation teaches about Qwen-like systems.

Pass condition:

- You can sample images through:

```text
noise -> latent generator -> latent -> decoder -> image
```

## Homework 6: Text-Conditioned Tiny Dataset

Question:

How does text become a useful condition for image generation?

Task:

- Create or use a tiny text-image dataset.
- Recommended dataset:
  - synthetic colored shapes
  - captions like "red circle", "blue square", "three green dots"
- Build:
  - tiny tokenizer
  - text embedding table
  - pooled text embedding
- Condition the latent/image generator on text.

Reading:

- Classifier-free guidance explanation.
- Cross-attention basics only if using cross-attention.

Deliverable:

- Dataset preview: image + caption pairs.
- Prompt-conditioned sample grid.
- Failure cases where prompt is ignored or partially followed.
- Comparison note: label conditioning vs text conditioning.

Pass condition:

- Changing the prompt changes the generated image in the expected direction.

## Homework 7: Classifier-Free Guidance

Question:

How can prompts become stronger during sampling?

Task:

- Add condition dropout during training.
- During sampling, run both:
  - conditional prediction
  - unconditional prediction
- Combine them with a guidance scale.
- Test multiple guidance scales.

Reading:

- Classifier-free guidance paper or a grounded explanation.
- Read only enough to connect to the two forward passes.

Deliverable:

- Same prompt sampled at several guidance scales.
- Notes on when guidance helps and when it breaks samples.
- Visual grid showing under-guidance, useful guidance, and over-guidance.

Pass condition:

- You can explain classifier-free guidance as "push away from unconditional generation toward prompt-conditioned generation."

## Homework 8: Tiny DiT

Question:

How does a transformer replace a U-Net in image generation?

Task:

- Patchify image latents into tokens.
- Add timestep embedding.
- Add text conditioning.
- Train a tiny DiT to predict noise or velocity.
- Unpatchify predictions back to latent/image shape.

Reading:

- DiT paper:
  - patch tokens
  - timestep conditioning
  - transformer denoiser
  - scaling intuition

Deliverable:

- Diagram of token shapes through the model.
- Sample grid from tiny text-conditioned DiT.
- Notes comparing DiT vs previous MLP/U-Net model.
- Source note comparing DiT implementations or notebooks inspected.

Pass condition:

- Your transformer model can generate simple prompt-conditioned images.

## Homework 9: Tiny MMDiT-Style Conditioning

Question:

How do text tokens and image tokens interact in a Qwen-like model?

Task:

- Stop using only pooled text embeddings.
- Keep text as a sequence of tokens.
- Keep image latents as a sequence of patch tokens.
- Build a tiny multimodal transformer:
  - text tokens attend with image tokens, or
  - separate streams with controlled interaction
- Predict noise or velocity for image tokens only.

Reading:

- Qwen-Image Technical Report:
  - MMDiT
  - text conditioning
  - model architecture
- Revisit DiT if token shapes are confusing.

Deliverable:

- Shape trace:

```text
caption tokens: [B, T_text, D]
image latent tokens: [B, T_img, D]
output image prediction: [B, T_img, D]
```

- Prompt-conditioned samples.
- Notes on how this is closer to Qwen-Image than the previous DiT.
- Comparison note: pooled text embedding vs token-level text conditioning.

Pass condition:

- You can point to:
  - text token stream
  - image token stream
  - multimodal transformer
  - image-token prediction head

## Homework 10: Final Project - Tiny Qwen-Like Text-To-Image

Question:

Can we assemble a tiny Qwen-like architecture end to end?

Task:

- Build the final tiny model:
  - synthetic captioned image dataset
  - tiny tokenizer
  - tiny text encoder
  - tiny autoencoder/VAE or simple learned latent space
  - tiny DiT/MMDiT denoiser or velocity model
  - diffusion or flow-matching training objective
  - classifier-free guidance
  - sampler
  - decoder

Recommended constraints:

- Images: 16x16 or 32x32.
- Dataset: colored shapes with captions.
- Text vocabulary: fewer than 100 tokens.
- Model: small enough to train/debug locally.
- Quality target: prompt following, not beauty.

Final deliverable:

- `final_project/` folder with runnable training and sampling scripts.
- README explaining:
  - architecture
  - tensor shapes
  - training objective
  - sampling loop
  - differences from real Qwen-Image
- Source map showing which notebooks, repos, and papers influenced each component.
- Ablation table:
  - no text conditioning
  - pooled text conditioning
  - token-level conditioning
  - low vs high guidance
- Sample grid for prompts:
  - "red circle"
  - "blue square"
  - "green triangle"
  - "two yellow circles"
  - deliberately hard prompts

Pass condition:

- Given a simple caption, the model generates an image that roughly matches the caption.
- You can explain how the final model maps:

```text
text + noise -> image
```

## Homework 11: Paper Mapping

Question:

Which parts of Qwen-Image are architecture, data, scale, or engineering?

Task:

- Read Qwen-Image Technical Report more carefully.
- Map each paper component to your tiny implementation:
  - same idea
  - simplified version
  - missing entirely
  - impossible without scale

Optional reading:

- Qwen-Image-VAE-2.0.
- Qwen-Image-2.0.

Deliverable:

- `notes/qwen_mapping.md`
- Table with columns:
  - Qwen component
  - tiny version
  - what was learned
  - what remains unknown

Pass condition:

- You understand the gap between a toy implementation and the real paper without treating the paper as magic.

## Suggested Repo Layout

```text
homeworks/
  hw0_system_map/
  hw1_2d_flow/
  hw2_2d_diffusion/
  hw3_tiny_images/
  hw4_autoencoder/
  hw5_latent_generation/
  hw6_text_conditioning/
  hw7_cfg/
  hw8_tiny_dit/
  hw9_tiny_mmdit/
  hw10_final_tiny_qwen/
notes/
  readings.md
  notebook_sources.md
  qwen_mapping.md
external/
  README.md
```

## Reading Priority

1. TorchCFM 2D examples.
2. Flow Matching for Generative Modeling.
3. DDPM.
4. Latent Diffusion.
5. Classifier-Free Guidance.
6. DiT.
7. Qwen-Image Technical Report.
8. Qwen-Image-VAE-2.0.
9. Qwen-Image-2.0.

Only read math deeply after you can point to the exact line of code it explains.
