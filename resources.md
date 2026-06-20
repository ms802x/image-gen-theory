# Public Resources For Diffusion And Flow Matching

This file tracks grounded resources to learn from. Prefer copying only from resources with explicit permissive licenses, and keep upstream attribution/license text when reusing code.

## Best Starting Points

### Flow matching

1. TorchCFM: Conditional Flow Matching
   - URL: https://github.com/atong01/conditional-flow-matching
   - License: MIT
   - Why it matters: has 2D tutorials, toy datasets, notebooks, and a real package implementing conditional flow matching variants.
   - Good first target: `examples/2D_tutorials/model-comparison-plotting.ipynb`
   - Reuse status: safe to copy/adapt with MIT attribution and license preservation.

2. facebookresearch/flow_matching
   - URL: https://github.com/facebookresearch/flow_matching
   - License: CC BY-NC 4.0
   - Why it matters: official-ish modern flow matching library tied to the Flow Matching Guide and Code paper.
   - Reuse status: fine for personal learning and non-commercial study, but do not treat it as fully permissive.

### Diffusion

3. Diffusion Explainer
   - URL: https://github.com/poloclub/diffusion-explainer
   - Live demo: https://poloclub.github.io/diffusion-explainer/
   - License: MIT
   - Why it matters: browser-based visual explanation of Stable Diffusion concepts. Useful before math.
   - Reuse status: safe to copy/adapt with MIT attribution and license preservation.

4. score_sde_pytorch
   - URL: https://github.com/yang-song/score_sde_pytorch
   - License: Apache-2.0
   - Why it matters: official PyTorch implementation for score-based generative modeling through SDEs. More advanced than the first toy experiments.
   - Reuse status: safe to copy/adapt with Apache-2.0 attribution and license preservation.

5. lucidrains/denoising-diffusion-pytorch
   - URL: https://github.com/lucidrains/denoising-diffusion-pytorch
   - License: MIT
   - Why it matters: widely used PyTorch DDPM implementation with clean package API.
   - Reuse status: safe to copy/adapt with MIT attribution and license preservation.

## Read, But Do Not Blindly Copy

6. Hugging Face Annotated Diffusion
   - URL: https://github.com/huggingface/blog/blob/main/annotated-diffusion.md
   - Why it matters: good line-by-line DDPM explanation.
   - License note: the blog repository page did not expose a clear repo license during this search.
   - Reuse status: read and link; do not copy substantial code/text unless license is clarified.

7. cloneofsimo/minDiffusion
   - URL: https://github.com/cloneofsimo/minDiffusion
   - Why it matters: very small educational DDPM implementation.
   - License note: no license file was found during this search.
   - Reuse status: read for intuition; do not copy code into this repo.

8. acids-ircam/diffusion_models
   - URL: https://github.com/acids-ircam/diffusion_models
   - Why it matters: tutorial notebooks on score matching, DDPM, WaveGrad, and DDIM.
   - License note: no clear license was visible during this search.
   - Reuse status: read for learning; do not copy code into this repo.

## Theory References To Connect After Code

1. DDPM paper
   - URL: https://arxiv.org/abs/2006.11239
   - Use after implementing or running a minimal diffusion loop.

2. Score-Based Generative Modeling through SDEs
   - URL: https://openreview.net/forum?id=PxTIG12RRHS
   - Use after understanding DDPM noise prediction and score matching.

3. Flow Matching for Generative Modeling
   - URL: https://arxiv.org/abs/2210.02747
   - Use after running a 2D flow matching example.

4. Flow Matching Guide and Code
   - URL: https://arxiv.org/abs/2412.06264
   - Use as the deeper reference once the TorchCFM examples make sense.

5. Introduction to Flow Matching and Diffusion Models
   - URL: https://arxiv.org/abs/2506.02070
   - Use as a bridge between code intuition and the broader MIT course-style theory.

## Recommended Learning Order

1. Run Diffusion Explainer in the browser.
2. Run TorchCFM's 2D tutorial and inspect the velocity field.
3. Read the Flow Matching paper only for the training objective and sampling picture.
4. Run a minimal DDPM implementation from a permissively licensed source.
5. Compare diffusion sampling steps vs. flow matching ODE steps on the same toy dataset.
6. Only then read SDE/score matching details.
