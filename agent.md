# Agent Instructions

This repo is for learning image generation theory through grounded, code-first experiments.

Current end goal:

- Build a tiny Qwen-Image-like text-to-image architecture for understanding.
- The near-term target is text-to-image, not image editing.
- The architecture should gradually work toward latent generation, text conditioning, and tiny DiT/MMDiT-style modeling.

User preferences:

- Prefer Karpathy-style learning: small, controlled experiments before heavy theory.
- Do not start with dense mathematical lectures or abstract derivations.
- Use CPU-friendly examples first.
- Focus first on 2D toy datasets, diffusion, and flow matching.
- Prefer existing public notebooks, repos, papers, and tutorials over creating notebooks from scratch.
- Ground recommendations in verifiable sources.
- Include useful public resources even when they are commercial, non-commercial, or have unclear licensing.
- Be careful with copying and reuse. Prefer resources with explicit permissive licenses when vendoring code, and keep attribution/license files when copying licensed code.
- Keep this folder linked to the personal GitHub repo only through its repo-local SSH key.
- Do not assume generated notebooks are correct without verification.
- Build intuition first, then connect each implementation detail back to the theory.

Current repo remote:

- `git@github.com:ms802x/image-gen-theory.git`

Repo-local SSH:

- This checkout uses a deploy key stored under `.git/ssh/`.
- Other folders should not use this key.
