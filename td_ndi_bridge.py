#!/usr/bin/env python3
"""Generate buffered frames from PM2.5 input and stream them over NDI.

This script keeps generation and playback separate:
- poll PM2.5 from OpenAQ or use a fixed manual value
- map PM2.5 to prompt structure controls
- generate a batch of LoRA keyframes when the data meaningfully changes
- interpolate those keyframes into a buffered frame sequence
- stream the buffered frames over NDI at a fixed playback fps

Source note:
- The NDI sender path in this file is adapted from the PyFAD repository and
  from Daniel's NDI examples / documentation used in the Programming for
  Artists and Designers course:
  https://github.com/colormotor/PyFAD.git

References:
- Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L.,
  & Chen, W. (2021). LoRA: Low-Rank Adaptation of Large Language Models.
  https://arxiv.org/abs/2106.09685
- Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
  High-Resolution Image Synthesis with Latent Diffusion Models.
  https://arxiv.org/abs/2112.10752
- Song, J., Meng, C., & Ermon, S. (2020). Denoising Diffusion Implicit Models.
  https://arxiv.org/abs/2010.02502
- Hugging Face Diffusers. Text-to-image training.
  https://huggingface.co/docs/diffusers/en/training/text2image
- Hugging Face Diffusers. LoRA loaders and inference.
  https://huggingface.co/docs/diffusers/en/api/loaders/lora
- OpenAQ Docs. About the API / API Key / Sensors / Latest.
  https://docs.openaq.org/about/about
  https://docs.openaq.org/using-the-api/api-key
  https://docs.openaq.org/api/operations/sensors_get_v3_locations__locations_id__sensors_get
  https://docs.openaq.org/api/operations/location_latest_get_v3_locations__locations_id__latest_get
- TouchDesigner Docs. NDI In TOP.
  https://docs.derivative.ca/NDI_In_TOP
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image


def resolve_project_root() -> Path:
    script_path = Path(__file__).resolve()
    markers = ("README.md", "environment.yml", "training_runs")

    for candidate in (script_path.parent, *script_path.parents):
        if all((candidate / marker).exists() for marker in markers):
            return candidate

    raise SystemExit(
        "Could not resolve the project root. Run this script from inside the repository "
        "or keep the repository files together."
    )


ROOT = resolve_project_root()
DEFAULT_LORA_DIR = ROOT / "training_runs" / "mycelium_lora_structure_v1"
DEFAULT_LORA_WEIGHT = "pytorch_lora_weights.safetensors"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "latent_exploration" / "ndi_live"
DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"

GUIDANCE_MIN = 6.9
GUIDANCE_MAX = 7.6
STEPS_MIN = 20
STEPS_MAX = 23


def load_project_env() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_project_env()


def env_float(name: str) -> Optional[float]:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    return float(value)


@dataclass
class MyceliumState:
    # Prompt-facing and TouchDesigner-facing controls derived from one PM2.5 reading.
    pm25: float
    normalized_pm25: float
    label: str
    density_level: float
    tangle_level: float
    density_phrase: str
    tangle_phrase: str
    structure_prompt: str
    stronger_prompt: str
    guidance_scale: float
    num_inference_steps: int
    td_density: float
    td_tangle: float
    td_guidance: float
    td_steps: float


@dataclass
class BufferedClip:
    # One fully prepared clip: source reading, generated keyframes, and playback frames.
    timestamp: str
    raw_pm25: float
    smoothed_pm25: float
    prompt: str
    state: MyceliumState
    keyframe_images: list[Image.Image]
    keyframe_frames: list[np.ndarray]
    keyframe_metadata: list[dict]
    playback_frames: list[np.ndarray]


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def lerp(start: float, end: float, t: float) -> float:
    return start + (end - start) * t


def normalize_range(value: float, start: float, end: float) -> float:
    if end == start:
        return 0.0
    return clamp01((value - start) / (end - start))


def resolve_device(preferred: str) -> str:
    if preferred == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    if preferred == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_openaq_pm25(api_key: str, location_id: Optional[int] = None) -> float:
    headers = {"X-API-Key": api_key}

    if location_id is not None:
        sensors_response = requests.get(
            f"https://api.openaq.org/v3/locations/{location_id}/sensors",
            headers=headers,
            timeout=20,
        )
        sensors_response.raise_for_status()
        sensors_payload = sensors_response.json()
        pm25_sensor_ids = {
            sensor["id"]
            for sensor in sensors_payload.get("results", [])
            if sensor.get("parameter", {}).get("name") == "pm25"
        }

        if not pm25_sensor_ids:
            raise RuntimeError(f"No PM2.5 sensors found for OpenAQ location {location_id}.")

        latest_response = requests.get(
            f"https://api.openaq.org/v3/locations/{location_id}/latest",
            headers=headers,
            timeout=20,
        )
        latest_response.raise_for_status()
        latest_payload = latest_response.json()

        for result in latest_payload.get("results", []):
            if result.get("sensorsId") in pm25_sensor_ids and result.get("value") is not None:
                return float(result["value"])

        raise RuntimeError(f"No PM2.5 latest value found for OpenAQ location {location_id}.")

    response = requests.get(
        "https://api.openaq.org/v3/latest",
        headers=headers,
        params={"parameter": "pm25", "limit": 1},
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()

    for result in payload.get("results", []):
        if result.get("value") is not None:
            return float(result["value"])

    raise RuntimeError("No PM2.5 reading found in the OpenAQ response.")


def density_phrase(level: float) -> str:
    if level < 0.2:
        return "very open branching density with long gaps between filaments"
    if level < 0.4:
        return "light branching density with separated filament paths"
    if level < 0.6:
        return "moderate branching density with more connected filament paths"
    if level < 0.8:
        return "dense branching density with tighter filament spacing"
    return "very dense branching density with compressed filament spacing"


def stronger_density_phrase(level: float) -> str:
    if level < 0.2:
        return "very sparse branching with strongly preserved negative space"
    if level < 0.4:
        return "light branching with clear spacing between thin filament bands"
    if level < 0.6:
        return "moderately packed branching with visibly connected filament lanes"
    if level < 0.8:
        return "dense interwoven branching with compact filament spacing"
    return "heavily packed branching mesh with strongly compressed filament spacing"


def tangle_phrase(level: float) -> str:
    if level < 0.2:
        return "low tangling with mostly clean directional traces"
    if level < 0.4:
        return "soft tangling with occasional filament crossings"
    if level < 0.6:
        return "moderate tangling with repeated overlapping turns"
    if level < 0.8:
        return "high tangling with compact overlapping crossings"
    return "heavy tangling with dense overlapping crossings"


def stronger_tangle_phrase(level: float) -> str:
    if level < 0.2:
        return "minimal tangling and restrained structural overlap"
    if level < 0.4:
        return "soft interweaving with lightly recurring crossings"
    if level < 0.6:
        return "noticeable interweaving with frequent structural overlaps"
    if level < 0.8:
        return "strong tangling with tight overlapping structural turns"
    return "very heavy tangling with saturated overlapping crossings"


def build_td_controls(state: MyceliumState) -> dict:
    return {
        "pm25_norm": round(state.normalized_pm25, 3),
        "density": round(state.td_density, 3),
        "tangle": round(state.td_tangle, 3),
        "guidance": round(state.td_guidance, 3),
        "steps": round(state.td_steps, 3),
    }


def map_pm25_to_state(pm25: float) -> MyceliumState:
    # Keep the environmental mapping in one place so prompt text and normalized TD controls
    # stay consistent whenever the incoming PM2.5 value changes.
    normalized = clamp01(pm25 / 60.0)
    density_level = clamp01(lerp(0.15, 0.95, normalized))
    tangle_level = clamp01(lerp(0.08, 0.88, normalized**1.15))

    if normalized < 0.2:
        label = "clean-air"
    elif normalized < 0.4:
        label = "moderate-air"
    elif normalized < 0.6:
        label = "elevated-air"
    elif normalized < 0.8:
        label = "unhealthy-sensitive"
    else:
        label = "unhealthy"

    current_density_phrase = density_phrase(density_level)
    current_tangle_phrase = tangle_phrase(tangle_level)
    current_stronger_density_phrase = stronger_density_phrase(density_level)
    current_stronger_tangle_phrase = stronger_tangle_phrase(tangle_level)

    structure_prompt = ", ".join(
        [
            "branching filament network",
            current_density_phrase,
            current_tangle_phrase,
        ]
    )
    stronger_prompt = ", ".join(
        [
            "branching filament network",
            current_stronger_density_phrase,
            current_stronger_tangle_phrase,
        ]
    )

    guidance_scale = round(lerp(GUIDANCE_MIN, GUIDANCE_MAX, normalized), 2)
    num_inference_steps = int(round(lerp(STEPS_MIN, STEPS_MAX, normalized)))
    td_guidance = normalize_range(guidance_scale, GUIDANCE_MIN, GUIDANCE_MAX)
    td_steps = normalize_range(num_inference_steps, STEPS_MIN, STEPS_MAX)

    return MyceliumState(
        pm25=pm25,
        normalized_pm25=round(normalized, 3),
        label=label,
        density_level=round(density_level, 3),
        tangle_level=round(tangle_level, 3),
        density_phrase=current_density_phrase,
        tangle_phrase=current_tangle_phrase,
        structure_prompt=structure_prompt,
        stronger_prompt=stronger_prompt,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        td_density=round(density_level, 3),
        td_tangle=round(tangle_level, 3),
        td_guidance=round(td_guidance, 3),
        td_steps=round(td_steps, 3),
    )


def build_prompt(pm25: float, variant: str = "structure_control", prefix: str = "mclm style"):
    state = map_pm25_to_state(pm25)
    prompt_body = state.structure_prompt if variant == "structure_control" else state.stronger_prompt
    return f"{prefix}, {prompt_body}", state


def smooth_value(current: float, previous: Optional[float], alpha: float) -> float:
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous


class NdiSender:
    """Small NDI sender wrapper for fixed-rate playback of buffered RGBA frames."""

    def __init__(self, name: str):
        try:
            import NDIlib as ndi
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "NDIlib is not available. Install ndi-python in the active environment."
            ) from exc

        self.ndi = ndi
        if not self.ndi.initialize():
            raise SystemExit("Could not initialize NDI.")

        create = self.ndi.SendCreate()
        create.ndi_name = name
        self.sender = self.ndi.send_create(create)
        if self.sender is None:
            raise SystemExit("Could not create NDI sender.")

    def send_rgba(self, frame: np.ndarray):
        # NDI expects a contiguous RGBA buffer plus explicit geometry metadata.
        frame = np.ascontiguousarray(frame, dtype=np.uint8)
        video_frame = self.ndi.VideoFrameV2()
        video_frame.data = frame
        video_frame.xres = frame.shape[1]
        video_frame.yres = frame.shape[0]
        video_frame.line_stride_in_bytes = frame.shape[1] * 4
        video_frame.FourCC = self.ndi.FOURCC_VIDEO_TYPE_RGBA
        self.ndi.send_send_video_v2(self.sender, video_frame)

    def close(self):
        self.ndi.send_destroy(self.sender)
        self.ndi.destroy()


class MyceliumGenerator:
    """Load the base model once and attach the local LoRA weights."""

    def __init__(self, model_id: str, lora_path: Path, lora_weight_name: str, lora_scale: float, device: str):
        self.model_id = model_id
        self.lora_path = lora_path
        self.lora_weight_name = lora_weight_name
        self.lora_scale = lora_scale
        self.device = resolve_device(device)
        self.pipe = None

    def load(self):
        if self.pipe is not None:
            return self.pipe

        from diffusers import StableDiffusionPipeline

        dtype = torch.float16 if self.device in {"cuda", "mps"} else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.device)

        if self.lora_path.exists():
            if self.lora_path.is_file():
                pipe.load_lora_weights(self.lora_path.parent, weight_name=self.lora_path.name)
            else:
                pipe.load_lora_weights(self.lora_path, weight_name=self.lora_weight_name)
            pipe.fuse_lora(lora_scale=self.lora_scale)

        self.pipe = pipe
        return pipe

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        num_inference_steps: int,
        guidance_scale: float,
        seed: int,
    ) -> Image.Image:
        pipe = self.load()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]


def save_json(path: Path, payload):
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_paths(output_dir: Path) -> dict[str, Path]:
    history_dir = output_dir / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "history_dir": history_dir,
        "latest_image": output_dir / "latest.png",
        "latest_metadata": output_dir / "latest.json",
        "latest_controls": output_dir / "latest_controls.json",
    }


def image_to_rgba_array(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGBA"), dtype=np.uint8)


def build_standby_frame(width: int, height: int) -> np.ndarray:
    frame = np.zeros((height, width, 4), dtype=np.uint8)
    frame[:, :, 3] = 255
    return frame


def blend_rgba_frames(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    blended = start.astype(np.float32) * (1.0 - alpha) + end.astype(np.float32) * alpha
    return np.clip(np.round(blended), 0, 255).astype(np.uint8)


def build_keyframe_times(duration: float, image_interval: float) -> list[float]:
    # These timestamps represent clip time, not wall time. They decide how many distinct
    # images we need to generate before we interpolate the in-between playback frames.
    times = [0.0]
    while times[-1] < duration:
        next_time = min(duration, times[-1] + image_interval)
        if next_time == times[-1]:
            break
        times.append(next_time)
    if len(times) == 1:
        times.append(duration)
    return times


def build_segment_frame_counts(total_frames: int, segment_count: int) -> list[int]:
    frame_budget = total_frames
    counts: list[int] = []
    for segment_index in range(segment_count):
        remaining_segments = segment_count - segment_index
        frames_for_segment = max(1, round(frame_budget / remaining_segments))
        counts.append(frames_for_segment)
        frame_budget -= frames_for_segment
    return counts


def build_playback_frames(keyframes: list[np.ndarray], duration: float, fps: float) -> list[np.ndarray]:
    if len(keyframes) == 1:
        return [keyframes[0]]

    # Expand a few expensive diffusion keyframes into a denser playback buffer. The sender
    # can then loop this list at a fixed fps without running generation work in the hot path.
    total_frames = max(1, int(round(duration * fps)))
    segment_counts = build_segment_frame_counts(total_frames, len(keyframes) - 1)
    playback_frames: list[np.ndarray] = []

    for segment_index, frame_count in enumerate(segment_counts):
        start_frame = keyframes[segment_index]
        end_frame = keyframes[segment_index + 1]
        for frame_index in range(frame_count):
            if segment_index < len(segment_counts) - 1:
                alpha = frame_index / max(frame_count, 1)
            else:
                alpha = 1.0 if frame_count == 1 else frame_index / max(frame_count - 1, 1)
            playback_frames.append(blend_rgba_frames(start_frame, end_frame, alpha))

    return playback_frames or [keyframes[-1]]


def make_keyframe_metadata(
    raw_pm25: float,
    smoothed_pm25: float,
    state: MyceliumState,
    prompt: str,
    args: argparse.Namespace,
    seed: int,
    keyframe_index: int,
    clip_time_seconds: float,
) -> dict:
    return {
        "keyframe_index": keyframe_index,
        "clip_time_seconds": round(clip_time_seconds, 3),
        "pm25_raw": raw_pm25,
        "pm25_smoothed": smoothed_pm25,
        "variant": args.variant,
        "prompt": prompt,
        "state": asdict(state),
        "td_controls": build_td_controls(state),
        "seed": seed,
        "width": args.width,
        "height": args.height,
        "num_inference_steps": state.num_inference_steps,
        "guidance_scale": state.guidance_scale,
        "model_id": args.model_id,
        "lora_path": str(args.lora_path),
        "lora_scale": args.lora_scale,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


def generate_buffered_clip(
    generator: MyceliumGenerator,
    args: argparse.Namespace,
    raw_pm25: float,
    smoothed_pm25: float,
) -> BufferedClip:
    # Clip generation is intentionally batch-oriented: once this returns, playback has
    # everything it needs and no longer depends on model inference timing.
    prompt, state = build_prompt(smoothed_pm25, variant=args.variant)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    keyframe_times = build_keyframe_times(args.duration, args.image_interval)
    keyframe_images: list[Image.Image] = []
    keyframe_frames: list[np.ndarray] = []
    keyframe_metadata: list[dict] = []

    for keyframe_index, clip_time_seconds in enumerate(keyframe_times):
        seed = args.base_seed + (keyframe_index * args.seed_step)
        image = generator.generate(
            prompt=prompt,
            width=args.width,
            height=args.height,
            num_inference_steps=state.num_inference_steps,
            guidance_scale=state.guidance_scale,
            seed=seed,
        )
        keyframe_images.append(image)
        keyframe_frames.append(image_to_rgba_array(image))
        keyframe_metadata.append(
            make_keyframe_metadata(
                raw_pm25=raw_pm25,
                smoothed_pm25=smoothed_pm25,
                state=state,
                prompt=prompt,
                args=args,
                seed=seed,
                keyframe_index=keyframe_index,
                clip_time_seconds=clip_time_seconds,
            )
        )

    return BufferedClip(
        timestamp=timestamp,
        raw_pm25=raw_pm25,
        smoothed_pm25=smoothed_pm25,
        prompt=prompt,
        state=state,
        keyframe_images=keyframe_images,
        keyframe_frames=keyframe_frames,
        keyframe_metadata=keyframe_metadata,
        playback_frames=build_playback_frames(keyframe_frames, args.duration, args.ndi_fps),
    )


def write_clip_outputs(paths: dict[str, Path], clip: BufferedClip, args: argparse.Namespace):
    # Persist the last generated visual state so TD-side tooling or manual inspection can
    # see what the current buffered clip was built from.
    keyframe_dir = paths["history_dir"] / f"{clip.timestamp}_keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)

    for metadata, image in zip(clip.keyframe_metadata, clip.keyframe_images):
        keyframe_path = keyframe_dir / f"keyframe_{metadata['keyframe_index']:03d}.png"
        image.save(keyframe_path)
        metadata["keyframe_path"] = str(keyframe_path)

    latest_image = clip.keyframe_images[-1]
    latest_image.save(paths["latest_image"])
    save_json(paths["latest_controls"], clip.keyframe_metadata[-1]["td_controls"])

    clip_metadata = {
        "timestamp": clip.timestamp,
        "raw_pm25": clip.raw_pm25,
        "smoothed_pm25": clip.smoothed_pm25,
        "label": clip.state.label,
        "prompt": clip.prompt,
        "duration_seconds": args.duration,
        "ndi_fps": args.ndi_fps,
        "image_interval_seconds": args.image_interval,
        "playback_frame_count": len(clip.playback_frames),
        "keyframe_count": len(clip.keyframe_metadata),
        "state": asdict(clip.state),
        "keyframes": clip.keyframe_metadata,
        "keyframe_dir": str(keyframe_dir),
    }

    save_json(paths["latest_metadata"], clip_metadata)
    save_json(paths["history_dir"] / f"{clip.timestamp}.json", clip_metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openaq-api-key", default=os.getenv("OPENAQ_API_KEY"))
    parser.add_argument("--openaq-location-id", type=int, default=os.getenv("OPENAQ_LOCATION_ID"))
    parser.add_argument(
        "--manual-pm25",
        type=float,
        default=env_float("MANUAL_PM25"),
        help="Use a fixed PM2.5 value instead of live API input.",
    )
    parser.add_argument("--variant", choices=["structure_control", "stronger_variants"], default="structure_control")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--lora-path", type=Path, default=DEFAULT_LORA_DIR)
    parser.add_argument("--lora-weight-name", default=DEFAULT_LORA_WEIGHT)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default=os.getenv("TD_NDI_DEVICE", "auto"))
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--seed-step", type=int, default=1, help="Seed increment between generated keyframes.")
    parser.add_argument("--duration", type=float, default=20.0, help="Buffered clip duration in seconds.")
    parser.add_argument(
        "--image-interval",
        type=float,
        default=8.0,
        help="Seconds of clip time represented by each generated keyframe.",
    )
    parser.add_argument("--poll-interval", type=float, default=60.0, help="Seconds between API polls.")
    parser.add_argument("--ndi-fps", type=float, default=2.0, help="Playback rate for buffered frames over NDI.")
    parser.add_argument("--ndi-name", default="Latent Mycelium NDI")
    parser.add_argument(
        "--change-threshold",
        type=float,
        default=5.0,
        help="Minimum PM2.5 delta required before regenerating the buffered clip.",
    )
    parser.add_argument(
        "--smoothing-alpha",
        type=float,
        default=0.35,
        help="Higher alpha follows the newest PM2.5 values more closely.",
    )
    parser.add_argument("--run-once", action="store_true", help="Generate one buffered clip, play it once, then exit.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace):
    if args.manual_pm25 is None and not args.openaq_api_key:
        raise SystemExit("Set OPENAQ_API_KEY or provide --manual-pm25.")
    if args.duration <= 0:
        raise SystemExit("--duration must be greater than 0.")
    if args.image_interval <= 0:
        raise SystemExit("--image-interval must be greater than 0.")
    if args.poll_interval <= 0:
        raise SystemExit("--poll-interval must be greater than 0.")
    if args.ndi_fps <= 0:
        raise SystemExit("--ndi-fps must be greater than 0.")


def fetch_pm25_sample(args: argparse.Namespace) -> float:
    if args.manual_pm25 is not None:
        return args.manual_pm25
    return get_openaq_pm25(args.openaq_api_key, args.openaq_location_id)


def should_regenerate_clip(
    smoothed_pm25: float,
    state_label: str,
    last_generated_pm25: Optional[float],
    last_generated_label: Optional[str],
    change_threshold: float,
) -> bool:
    # Ignore small sensor jitter, but regenerate immediately when the qualitative state
    # category changes even if the numeric delta is small.
    if last_generated_pm25 is None or last_generated_label is None:
        return True
    if state_label != last_generated_label:
        return True
    return abs(smoothed_pm25 - last_generated_pm25) >= change_threshold


def main():
    args = parse_args()
    validate_args(args)

    paths = build_paths(args.output_dir)
    ndi_sender = NdiSender(args.ndi_name)
    standby_frame = build_standby_frame(args.width, args.height)
    generator = MyceliumGenerator(
        model_id=args.model_id,
        lora_path=args.lora_path,
        lora_weight_name=args.lora_weight_name,
        lora_scale=args.lora_scale,
        device=args.device,
    )
    executor = ThreadPoolExecutor(max_workers=1)

    current_clip: Optional[BufferedClip] = None
    pending_clip: Optional[Future[BufferedClip]] = None
    previous_pm25 = None
    last_generated_pm25: Optional[float] = None
    last_generated_label: Optional[str] = None
    playback_index = 0
    next_frame_time = 0.0
    next_poll_time = 0.0
    sleep_interval = min(0.02, 1.0 / max(args.ndi_fps, 1.0))
    played_once = False

    try:
        while True:
            now = time.time()

            if pending_clip is not None and pending_clip.done():
                # Swap in the new buffer atomically once generation finishes.
                current_clip = pending_clip.result()
                write_clip_outputs(paths, current_clip, args)
                last_generated_pm25 = current_clip.smoothed_pm25
                last_generated_label = current_clip.state.label
                playback_index = 0
                pending_clip = None
                played_once = False

                print(
                    json.dumps(
                        {
                            "event": "clip_ready",
                            "timestamp": current_clip.timestamp,
                            "pm25_smoothed": current_clip.smoothed_pm25,
                            "label": current_clip.state.label,
                            "playback_frame_count": len(current_clip.playback_frames),
                            "keyframe_count": len(current_clip.keyframe_metadata),
                        },
                        indent=2,
                    )
                )

            if now >= next_poll_time:
                # Polling and generation decisions happen on a slower cadence than playback.
                raw_pm25 = fetch_pm25_sample(args)
                smoothed_pm25 = round(smooth_value(raw_pm25, previous_pm25, args.smoothing_alpha), 2)
                previous_pm25 = smoothed_pm25
                _, state = build_prompt(smoothed_pm25, variant=args.variant)

                regenerate = (
                    pending_clip is None
                    and should_regenerate_clip(
                        smoothed_pm25=smoothed_pm25,
                        state_label=state.label,
                        last_generated_pm25=last_generated_pm25,
                        last_generated_label=last_generated_label,
                        change_threshold=args.change_threshold,
                    )
                )

                if regenerate:
                    # Generation runs off the playback path so NDI can keep streaming the
                    # previous buffer while the next one is being prepared.
                    pending_clip = executor.submit(
                        generate_buffered_clip,
                        generator,
                        args,
                        raw_pm25,
                        smoothed_pm25,
                    )

                print(
                    json.dumps(
                        {
                            "event": "poll",
                            "pm25_raw": raw_pm25,
                            "pm25_smoothed": smoothed_pm25,
                            "label": state.label,
                            "regenerating": bool(regenerate),
                            "generation_in_progress": pending_clip is not None,
                        },
                        indent=2,
                    )
                )

                next_poll_time = now + args.poll_interval

            if now >= next_frame_time:
                # Send a standby frame immediately on startup so TD can discover the NDI source
                # even while the first diffusion clip is still generating.
                if current_clip is None:
                    ndi_sender.send_rgba(standby_frame)
                else:
                    # Playback is intentionally simple: send the next buffered frame, wrap, repeat.
                    ndi_sender.send_rgba(current_clip.playback_frames[playback_index])
                    playback_index += 1

                    if playback_index >= len(current_clip.playback_frames):
                        playback_index = 0
                        played_once = True

                next_frame_time = now + (1.0 / args.ndi_fps)

                if args.run_once and played_once and pending_clip is None:
                    break

            time.sleep(sleep_interval)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        ndi_sender.close()


if __name__ == "__main__":
    main()
