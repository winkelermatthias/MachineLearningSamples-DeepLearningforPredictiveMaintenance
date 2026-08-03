"""Run configuration, API keys and the teacher registry."""
import os
from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 20816            # the ISO standard number, why not
    n_samples: int = 40_000      # synthetic spectra (cached to disk after first run)
    n_teacher: int = 4_000       # samples sent to the teachers  <-- main API cost/time knob
    n_eval: int = 300            # held-out ground-truth eval set (teachers never see it)
    eval_run: int = 150          # eval prompts actually generated per model
    blind_fraction: float = 0.25  # fraction where the bearing must be identified
    epochs: int = 3
    out_dir: str = "/content/distill_pdm"

    def __post_init__(self):
        os.makedirs(self.out_dir, exist_ok=True)


def get_secret(name):
    """Colab secret (key icon in the sidebar), env var, or interactive prompt."""
    try:
        from google.colab import userdata
        v = userdata.get(name)
        if v:
            return v
    except Exception:
        pass
    if os.environ.get(name):
        return os.environ[name]
    from getpass import getpass
    return getpass(f"{name}: ")


def teacher_registry(nvidia_api_key, moonshot_api_key):
    return {
        "glm-5.2": dict(
            base_url="https://integrate.api.nvidia.com/v1",  # NVIDIA NIM, OpenAI-compatible
            api_key=nvidia_api_key,
            model="z-ai/glm-5.2",
            concurrency=6,   # NIM free tier is rate-limited; raise if you have credits
        ),
        "kimi-k3": dict(
            base_url="https://api.moonshot.ai/v1",           # Moonshot, OpenAI-compatible
            api_key=moonshot_api_key,
            model="kimi-k3",  # 'kimi-latest' was retired Jan 2026 -> K3 is the flagship
            concurrency=4,
        ),
    }
