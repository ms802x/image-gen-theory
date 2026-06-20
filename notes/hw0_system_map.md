# HW0 System Map

Question: what pieces does a tiny Qwen-like text-to-image system need?

This note maps the real target system into a tiny learning version.

## Real System Shape

Modern text-to-image systems usually look like this:

```text
text prompt
  -> tokenizer
  -> text encoder
  -> text condition tokens

random noise
  -> image latent tokens
  -> diffusion/flow transformer conditioned on text
  -> denoised/generated latent tokens
  -> VAE decoder
  -> image
```

Qwen-like systems add scale and engineering around this:

- strong multilingual text understanding
- high-quality VAE latents
- DiT/MMDiT-style transformer backbone
- large and carefully filtered text-image data
- training curriculum
- special attention to text rendering and prompt following

## Tiny Version

The tiny version should keep the core ideas but remove scale:

```text
caption
  -> tiny tokenizer
  -> tiny text encoder
  -> text tokens or pooled text embedding

noise latent
  -> patch tokens
  -> tiny DiT/MMDiT
  -> predicted noise or velocity
  -> sampler loop
  -> generated latent
  -> tiny decoder
  -> 16x16 or 32x32 image
```

Recommended first data:

```text
"red circle"       -> 32x32 image with red circle
"blue square"      -> 32x32 image with blue square
"two green dots"   -> 32x32 image with two green dots
"yellow triangle"  -> 32x32 image with yellow triangle
```

## Component Map

| Component | Real Qwen-like version | Tiny homework version | What it teaches |
| --- | --- | --- | --- |
| Tokenizer | large multilingual tokenizer | tiny word-level tokenizer | text must become tokens/ids before conditioning |
| Text encoder | large language/VL model features | embedding table or tiny transformer | prompt information becomes vectors |
| Image representation | high-quality VAE latent | tiny autoencoder/VAE latent or patchified pixels | generation often happens outside pixel space |
| Noise process | diffusion or flow in latent space | 2D, pixel, then latent noise | generator learns a path from noise to data |
| Backbone | DiT/MMDiT | tiny transformer | images can be treated as token sequences |
| Conditioning | text/image token interaction | pooled text first, token-level later | prompt following depends on conditioning strength |
| Sampler | many optimized steps | simple Euler or DDPM sampler | generation is an iterative process |
| Decoder | production VAE decoder | tiny decoder | latent quality limits output quality |

## Main Comparisons To Understand

Diffusion vs flow matching:

- Diffusion learns denoising.
- Flow matching learns velocity/motion.
- Both can turn noise into samples, but the training target and sampler feel different.

Pixel vs latent generation:

- Pixel generation is conceptually simple.
- Latent generation is cheaper and closer to modern systems.
- Bad latents make generation worse no matter how good the generator is.

U-Net vs DiT:

- U-Nets bake in image locality through convolutions.
- DiTs treat image or latent patches as tokens.
- DiTs become more natural when text and image are both token sequences.

Pooled text vs token-level text:

- Pooled text is simple and good for toy prompts.
- Token-level conditioning is closer to real text-to-image systems.
- Long prompts, text rendering, and spatial relations need more than one pooled vector.

## Tiny Final Target

Minimum final project:

```text
synthetic captioned image dataset
+ tiny tokenizer
+ tiny text encoder
+ tiny autoencoder/VAE or direct latent representation
+ tiny DiT/MMDiT
+ diffusion or flow objective
+ classifier-free guidance
+ sampler
+ decoder
```

Success means:

- simple prompts change the generated image
- the model can overfit a tiny dataset
- each tensor shape is explainable
- each piece maps to a Qwen-like component
- failures are understood instead of hidden

## What Not To Attempt Early

- high-resolution images
- photorealism
- OCR-quality text rendering
- huge datasets
- training a production VAE
- copying a large repo before reducing the idea

The purpose of the homeworks is to make the real system less mysterious one part at a time.
