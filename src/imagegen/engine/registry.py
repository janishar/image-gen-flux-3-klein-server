"""The lazily loaded, resident pipeline and prompt upsampler."""

from __future__ import annotations

import torch

from ..schemas import MODELS, GenerateRequest
from . import klein
from .device import patch_mps_empty_cache


class ModelRegistry:
    """Loads Klein on first use, and the upsampler on the first request for it, and keeps them until the process exits.

    ponytail: both resident ≈39 GB in bf16 of a 64 GB Mac; no eviction, add one if a smaller machine needs the upsampler.
    """

    def __init__(self, model_paths: dict[str, str], device: str, upsampler_path: str | None = None,
                 transformer: str = "bf16", text_encoder: str = "bf16", memory_fraction: float = 0):
        self.model_paths = dict(model_paths)
        self.device = device
        self.upsampler_path = upsampler_path
        self.precision = {"transformer": transformer, "text_encoder": text_encoder}
        self._pipes: dict[str, object] = {}
        self._upsampler = None
        if device == "mps":
            patch_mps_empty_cache()
        if device == "cuda" and memory_fraction:  # an overrun fails this render, not vLLM beside it
            torch.cuda.set_per_process_memory_fraction(memory_fraction)

    @property
    def configured(self) -> tuple[str, ...]:
        return tuple(name for name in MODELS if name in self.model_paths)

    @property
    def can_upsample(self) -> bool:
        return self.upsampler_path is not None

    def status(self) -> dict[str, dict]:
        return {
            name: {"configured": name in self.model_paths, "loaded": name in self._pipes}
            for name in MODELS
        }

    def upsampler_status(self) -> dict:
        return {"configured": self.can_upsample, "loaded": self._upsampler is not None}

    def get(self, name: str):
        """The pipeline for `name`, loading it now if needed. Worker thread only."""
        if name not in self._pipes:
            print(f"Loading {name} from {self.model_paths[name]}", flush=True)
            self._pipes[name] = klein.load_pipeline(self.model_paths[name], self.device, **self.precision)
        return self._pipes[name]

    def upsampler(self) -> tuple:
        """The upsampler, loading it now if needed. Worker thread only."""
        if self._upsampler is None:
            print(f"Loading the upsampler from {self.upsampler_path}", flush=True)
            self._upsampler = klein.load_upsampler(self.upsampler_path, self.device)
        return self._upsampler

    def render(self, req: GenerateRequest, on_step=None):
        try:
            return klein.render(self.get(req.model), self.upsampler() if req.upsample else None, req, on_step)
        finally:
            if self.device == "cuda":  # a vLLM restart needs this memory free
                torch.cuda.empty_cache()
