# image-gen-theory

Small, controlled experiments and grounded public resources for learning image generation theory from code first.

The goal is to build intuition before reading the full mathematical theory, then use that understanding to build a tiny Qwen-Image-like text-to-image architecture.

Start with the homework sequence in [ROADMAP.md](ROADMAP.md).

## Focus

- 2D toy datasets
- diffusion from scratch
- flow matching from scratch
- tiny text-to-image
- tiny DiT / MMDiT-style architecture
- latent image generation
- CPU-friendly PyTorch experiments
- plots and animations that make the model behavior visible
- public notebooks, repos, papers, and demos before generated notebooks

## Resource Map

| Resource | Topic | Type | License / reuse note | Best use |
| --- | --- | --- | --- | --- |
| [TorchCFM](https://github.com/atong01/conditional-flow-matching) | Flow matching | PyTorch library + notebooks | MIT | Start here for 2D flow matching tutorials. |
| [facebookresearch/flow_matching](https://github.com/facebookresearch/flow_matching) | Flow matching | PyTorch library + examples | CC BY-NC 4.0 | Strong modern reference for non-commercial learning. |
| [Diffusion Explainer](https://github.com/poloclub/diffusion-explainer) | Diffusion / Stable Diffusion | Browser visualization | MIT | Visual intuition before math. |
| [Diffusion Explainer live demo](https://poloclub.github.io/diffusion-explainer/) | Diffusion / Stable Diffusion | Interactive website | MIT project | First thing to play with. |
| [score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) | Score-based diffusion / SDEs | Official PyTorch implementation | Apache-2.0 | Advanced reference after DDPM basics. |
| [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) | DDPM | PyTorch package | MIT | Practical diffusion implementation to inspect. |
| [Hugging Face Annotated Diffusion](https://github.com/huggingface/blog/blob/main/annotated-diffusion.md) | DDPM | Annotated article/code | License unclear in search | Read and link; verify before copying. |
| [minDiffusion](https://github.com/cloneofsimo/minDiffusion) | DDPM | Minimal PyTorch implementation | No clear license found | Read for intuition; verify before copying. |
| [acids-ircam/diffusion_models](https://github.com/acids-ircam/diffusion_models) | DDPM / score matching / DDIM | Tutorial notebooks | No clear license found | Read for learning; verify before copying. |
| [DDPM paper](https://arxiv.org/abs/2006.11239) | Diffusion theory | Paper | Public paper | Read after running a small DDPM. |
| [Score-Based Generative Modeling through SDEs](https://openreview.net/forum?id=PxTIG12RRHS) | Score matching / SDEs | Paper | Public paper | Read after noise prediction and scores click. |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | Flow matching theory | Paper | Public paper | Read after a 2D flow example. |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) | Flow matching | Paper + codebase reference | Public paper | Deeper reference after TorchCFM. |
| [Introduction to Flow Matching and Diffusion Models](https://arxiv.org/abs/2506.02070) | Diffusion + flow matching | Course-style notes | Public paper | Bridge from code intuition to theory. |

## Learning Order

1. Run Diffusion Explainer in the browser.
2. Run TorchCFM's 2D tutorial and inspect the learned velocity field.
3. Read only the practical parts of the Flow Matching paper.
4. Run or inspect a minimal DDPM from a public source.
5. Compare diffusion sampling steps against flow matching ODE steps.
6. Then read score matching, SDEs, and the heavier theory.

## Notes

Commercial, non-commercial, and unclear-license resources can all be useful for personal learning. For copying code into this repo, keep attribution and preserve upstream license files when a license exists; verify unclear-license sources before vendoring them.
