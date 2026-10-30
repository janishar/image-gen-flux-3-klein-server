"""The /generate request: its fields, defaults and validation."""

from __future__ import annotations

import base64
import binascii
import io
import struct
from collections.abc import Collection
from dataclasses import dataclass, fields, replace

from PIL import Image

ASPECT_RATIOS = {
    "1:1": (1024, 1024),
    "4:3": (1152, 864),
    "3:4": (864, 1152),
    "3:2": (1248, 832),
    "2:3": (832, 1248),
    "16:9": (1360, 768),
    "9:16": (768, 1360),
}

MODELS = ("klein",)
MAX_IMAGES = 10
MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE = 64, 4096, 16
MAX_INPUT_PIXELS = 2 * MAX_SIDE * MAX_SIDE
MAX_STEPS = 16  # the distilled checkpoint's schedule


class ValidationError(ValueError):
    pass


@dataclass(frozen=True)
class GenerateRequest:
    prompt: str
    model: str = "klein"
    ratio: str | None = None
    width: int | None = None
    height: int | None = None
    steps: int = MAX_STEPS
    seed: int = 42
    upsample: bool = False
    images: tuple = ()  # raw image bytes once parsed (small while queued); PIL images after load_images


_KEYS = {f.name for f in fields(GenerateRequest)}


def _int(data: dict, key: str, lo: int, hi: int, multiple: int = 1) -> int | None:
    value = data.get(key)
    if value is None:
        return None
    if type(value) is not int or not lo <= value <= hi or value % multiple:
        extra = f", a multiple of {multiple}" if multiple > 1 else ""
        raise ValidationError(f"{key} must be an integer in {lo}..{hi}{extra}")
    return value


def _decode_image(i: int, text: object) -> bytes:
    if not isinstance(text, str):
        raise ValidationError(f"images[{i}] must be a base64 string")
    try:
        raw = base64.b64decode(text, validate=True)
        img = Image.open(io.BytesIO(raw))
        if img.width * img.height > MAX_INPUT_PIXELS:
            raise ValidationError(f"images[{i}] is {img.width}x{img.height}, over {MAX_INPUT_PIXELS} pixels")
        img.load()  # decoded only to validate; the worker decodes again in load_images
    except ValidationError:
        raise
    except (binascii.Error, OSError, ValueError, SyntaxError, IndexError, TypeError, struct.error,
            Image.DecompressionBombError) as e:
        raise ValidationError(f"images[{i}] is not a valid image: {e}") from None
    return raw


def load_images(req: GenerateRequest) -> GenerateRequest:
    """`req` with its validated image bytes decoded, for the render."""
    images = []
    for raw in req.images:
        img = Image.open(io.BytesIO(raw))
        img.load()
        images.append(img)
    return replace(req, images=tuple(images))


def dump_request(req: GenerateRequest) -> dict:
    """`req` as a JSON-able dict that parse_request turns back into it (images as base64)."""
    data = {f.name: getattr(req, f.name) for f in fields(GenerateRequest)}
    data["images"] = [base64.b64encode(raw).decode() for raw in req.images]
    return data


def parse_request(data: object, available_models: Collection[str], can_upsample: bool = True) -> GenerateRequest:
    """Check a decoded JSON body and turn it into a GenerateRequest, raising ValidationError on anything wrong."""
    if not isinstance(data, dict):
        raise ValidationError("request body must be a JSON object")
    unknown = sorted(set(data) - _KEYS)
    if unknown:
        raise ValidationError(f"unknown field(s): {', '.join(unknown)}")

    prompt = data.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValidationError("prompt must be a non-empty string")

    model = data.get("model", MODELS[0])
    if model not in MODELS:
        raise ValidationError(f"model must be one of {', '.join(MODELS)}")
    if model not in available_models:
        raise ValidationError(f"model {model!r} is not configured")

    ratio = data.get("ratio")
    if ratio is not None and (not isinstance(ratio, str) or ratio not in ASPECT_RATIOS):
        raise ValidationError(f"ratio must be one of {', '.join(ASPECT_RATIOS)}")

    upsample = data.get("upsample", False)
    if not isinstance(upsample, bool):
        raise ValidationError("upsample must be a boolean")
    if upsample and not can_upsample:
        raise ValidationError("the prompt upsampler is not configured")

    images = data.get("images", [])
    if not isinstance(images, list):
        raise ValidationError("images must be a list of base64 strings")
    if len(images) > MAX_IMAGES:
        raise ValidationError(f"at most {MAX_IMAGES} images")

    steps = _int(data, "steps", 1, 100)
    seed = _int(data, "seed", 0, 2**63 - 1)
    return GenerateRequest(
        prompt=prompt,
        model=model,
        ratio=ratio,
        width=_int(data, "width", MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE),
        height=_int(data, "height", MIN_SIDE, MAX_SIDE, SIDE_MULTIPLE),
        steps=MAX_STEPS if steps is None else min(steps, MAX_STEPS),  # Qwen clients may send 40
        seed=42 if seed is None else seed,
        upsample=upsample,
        images=tuple(_decode_image(i, text) for i, text in enumerate(images)),
    )
