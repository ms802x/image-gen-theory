import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JUPYTER = ROOT / ".miniforge/bin/jupyter"
NOTEBOOKS = [
    ROOT / "homeworks/hw0_system_map/hw0_system_map.ipynb",
    ROOT / "homeworks/hw1_2d_flow/hw1_2d_flow.ipynb",
    ROOT / "homeworks/hw2_2d_diffusion/hw2_2d_diffusion.ipynb",
    ROOT / "homeworks/hw3_tiny_images/hw3_tiny_images.ipynb",
    ROOT / "homeworks/hw4_autoencoder_vae/hw4_autoencoder_vae.ipynb",
    ROOT / "homeworks/hw5_latent_generation/hw5_latent_generation.ipynb",
    ROOT / "homeworks/hw6_text_conditioning/hw6_text_conditioning.ipynb",
    ROOT / "homeworks/hw7_cfg/hw7_cfg.ipynb",
    ROOT / "homeworks/hw8_tiny_dit/hw8_tiny_dit.ipynb",
    ROOT / "homeworks/hw9_tiny_mmdit/hw9_tiny_mmdit.ipynb",
    ROOT / "homeworks/hw10_final_tiny_qwen/hw10_final_tiny_qwen.ipynb",
    ROOT / "homeworks/hw11_qwen_mapping/hw11_qwen_mapping.ipynb",
]


def main():
    env = os.environ.copy()
    env["MPLBACKEND"] = "Agg"
    for path in NOTEBOOKS:
        start = time.time()
        rel = path.relative_to(ROOT)
        print(f"executing {rel}", flush=True)
        result = subprocess.run(
            [
                str(JUPYTER),
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=900",
                str(path),
            ],
            cwd=ROOT,
            env=env,
        )
        elapsed = time.time() - start
        if result.returncode != 0:
            print(f"failed {rel} after {elapsed:.1f}s", flush=True)
            return result.returncode
        print(f"done {rel} in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
