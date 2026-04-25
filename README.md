# The Latent Mycelium

The Latent Mycelium is a macOS-based art and ML project that combines LoRA training, realtime image generation, and TouchDesigner bridge scripts driven by environmental data.

## Setup

These instructions are written for an Apple Silicon Mac.

### Prerequisites

Install these first:

- Git
- Conda or Miniforge
- `mamba` in your base Conda environment

If `mamba` is not installed yet:

```bash
conda install -n base -c conda-forge mamba
```

### 1. Clone the repository

```bash
git clone https://github.com/<your-user-or-org>/The-Latent-Mycelium.git
cd The-Latent-Mycelium
```

## Project Scope

This repository currently includes:

- LoRA training assets and notebooks for mycelium image generation
- a maintained Conda environment for the main ML workflow
- a buffered NDI workflow for TouchDesigner playback
- dataset captions and metadata used for training and prompt control

## Repository Layout

- [environment.yml](environment.yml): maintained Conda environment for the main project workflow
- [ml-art-environment.yml](ml-art-environment.yml): exported snapshot of a known working environment
- [train_text_to_image_lora.py](train_text_to_image_lora.py): LoRA training script
- [td_ndi_bridge.py](td_ndi_bridge.py): buffered NDI renderer and sender for TouchDesigner playback
- [dataset](dataset): training images, captions, and metadata
- [training_runs](training_runs): trained LoRA outputs

`td_ndi_bridge.py` now lives in the repository root so it can be run directly without a `scripts/` prefix.

## Environment Strategy

This repo uses two Python environments:

1. `ml-art`
   Used for notebooks, LoRA training, diffusers inference, MediaPipe, and OpenCV.
2. `ml-art-ndi`
   Used for [td_ndi_bridge.py](td_ndi_bridge.py) and NDI output to TouchDesigner.

The split matters because `ndi-python` works cleanly in a separate Python 3.10 environment, while the main project environment is maintained on Python 3.11.

### 2. Install the environments

Install `ml-art` first:

```bash
mamba env create -f environment.yml
conda activate ml-art
python -m ipykernel install --user --name ml-art --display-name "Python (ml-art)"
```

If `ml-art` already exists:

```bash
mamba env update -f environment.yml
conda activate ml-art
```

Verify `ml-art`:

```bash
python -c "import torch; print(torch.__version__); print(torch.backends.mps.is_available()); print(torch.backends.mps.is_built())"
python -c "import mediapipe as mp; print(mp.__version__)"
python -c "import diffusers, transformers, accelerate, safetensors; print(diffusers.__version__); print(transformers.__version__); print(accelerate.__version__)"
python -c "import websockets; print(websockets.__version__)"
python -m jupyter kernelspec list
```

Expected:

- `torch.backends.mps.is_available()` should print `True` on a compatible Apple Silicon Mac
- Jupyter should list `ml-art`

Install `ml-art-ndi` next:

```bash
conda create -n ml-art-ndi python=3.10 -y
conda activate ml-art-ndi
python -m pip install requests pillow torch torchvision diffusers transformers accelerate safetensors peft
python -m pip install ndi-python==5.1.1.5
```

Verify `ml-art-ndi`:

```bash
python -c "import NDIlib, requests, torch; from PIL import Image; import diffusers, transformers, accelerate, safetensors, peft, torchvision; print('buffered ndi env ok')"
```

Expected:

- `buffered ndi env ok` prints

## Terminal Workflow: NDI to TouchDesigner

The current TouchDesigner path in this repository is buffered NDI playback.

### 3. Run the sender from the terminal

Important:

- Use `ml-art-ndi`, not `ml-art`
- Run the command from the repository root
- If `--device mps` does not work on your machine, switch it to `--device cpu`

Basic manual test command:

```bash
conda activate ml-art-ndi
python td_ndi_bridge.py --device mps --manual-pm25 25 --duration 20 --image-interval 8 --ndi-fps 6 --change-threshold 5 --poll-interval 60
```

What this command does:

- starts the NDI sender named `Latent Mycelium NDI`
- generates buffered keyframes from a fixed PM2.5 value of `25`
- loops the buffered frames over NDI for TouchDesigner

If you want to use live OpenAQ data instead of a fixed manual value:

```bash
export OPENAQ_API_KEY="your_api_key_here"
export OPENAQ_LOCATION_ID="your_location_id_here"
conda activate ml-art-ndi
python td_ndi_bridge.py --device mps --duration 20 --image-interval 8 --ndi-fps 6 --change-threshold 5 --poll-interval 60
```

### 4. Receive the stream in TouchDesigner

In TouchDesigner:

1. Add an `NDI In TOP`
2. Open its source/device menu
3. Select `Latent Mycelium NDI`

If you override `--ndi-name`, select that custom name instead.

### 5. Check the output files

While the sender runs, it writes:

- `outputs/latent_exploration/ndi_live/latest.json`
- `outputs/latent_exploration/ndi_live/latest.png`
- `outputs/latent_exploration/ndi_live/latest_controls.json`
- timestamped metadata and keyframes in `outputs/latent_exploration/ndi_live/history/`

### Common parameters

- `--device mps`: use Apple Metal on Apple Silicon
- `--device cpu`: safer fallback if MPS is unavailable
- `--manual-pm25 25`: use a fixed PM2.5 value for testing
- `--duration 20`: buffered clip length in seconds
- `--image-interval 8`: seconds of clip time between generated keyframes
- `--ndi-fps 6`: playback fps for the buffered frames
- `--change-threshold 5`: regenerate only when PM2.5 changes enough
- `--poll-interval 60`: poll live data every 60 seconds

## VS Code and Jupyter

For notebooks and training work:

1. Open the project in VS Code.
2. Select the interpreter or kernel named `Python (ml-art)`.
3. Keep `ml-art-ndi` for terminal-based buffered NDI runs unless you explicitly need a separate kernel for it.

## Snapshot Environment

This repository also includes a full exported environment snapshot:

- [ml-art-environment.yml](ml-art-environment.yml)

Use it only when you specifically want to recreate that exact exported state:

```bash
mamba env create -f ml-art-environment.yml
```

For day-to-day maintenance:

- edit [environment.yml](environment.yml)
- use `mamba env create -f environment.yml` or `mamba env update -f environment.yml`
- keep [ml-art-environment.yml](ml-art-environment.yml) as an exported snapshot, not the primary hand-edited file
