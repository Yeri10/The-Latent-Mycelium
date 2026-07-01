# The Latent Mycelium

The Latent Mycelium is a macOS-based art and ML project that combines LoRA image generation, PM2.5-driven prompt control, and TouchDesigner playback through NDI.

The main live-performance workflow is:

1. Run `td_ndi_bridge.py` from Terminal.
2. The script generates mycelium images from a fixed or live PM2.5 value.
3. The script sends the generated frames as an NDI stream.
4. TouchDesigner receives the stream through an `NDI In TOP`.

## Run This Project

Start with **Quick Run** if you want to test the system with a fixed PM2.5 value. This is the safest first step.

Use **Live OpenAQ Data** if you want to run the project with realtime PM2.5 data from the OpenAQ API. In live mode, do not include `--manual-pm25`; the script reads `OPENAQ_API_KEY` and `OPENAQ_LOCATION_ID` from the project `.env` file.

After the Python sender is running, open the TouchDesigner file and follow **TouchDesigner** to receive the `Latent Mycelium NDI` stream.

## Run On Another Computer

If you want to run this project on another computer, move or clone the entire repository, not just the Python scripts. The live sender expects the repository structure to stay intact because it resolves the project root from `README.md`, `environment.yml`, and `training_runs`, and it loads the LoRA weights from `training_runs/mycelium_lora_structure_v1`.

Before the first run on a new machine:

1. Install Conda or Miniforge.
2. Copy or clone this whole repository to the new computer.
3. Decide whether you need only the live NDI workflow or also notebooks/training.
4. Create the `ml-art-ndi` environment first because it is the smallest path to a successful test.
5. Stay online for the first inference run so Diffusers can download the base model `runwayml/stable-diffusion-v1-5`.

Use `ml-art-ndi` if your goal is to generate images and send them to TouchDesigner over NDI.

Use `ml-art` only if you also need notebooks, experiments, or LoRA training.

## Quick Run

Use this section if the Conda environments already exist.

### 1. Open the repository

```bash
cd /path/to/The-Latent-Mycelium
```

If the repository is cloned somewhere else, replace the path with your local project path.

### 2. Activate the NDI environment

```bash
conda activate ml-art-ndi
```

The terminal prompt should begin with:

```text
(ml-art-ndi)
```

### 3. Run a short test

This command generates one short buffered clip and then exits. It is the best first check.

```bash
python td_ndi_bridge.py --device cpu --manual-pm25 25 --duration 8 --image-interval 4 --ndi-fps 2 --run-once
```

The first run can take several minutes because PyTorch, Diffusers, the base Stable Diffusion model, and the LoRA weights need to load.

Expected terminal output includes JSON messages such as:

```text
"event": "poll"
"regenerating": true
"event": "clip_ready"
```

When this succeeds, the script writes files to:

```text
outputs/latent_exploration/ndi_live/
```

The most useful preview file is:

```text
outputs/latent_exploration/ndi_live/latest.png
```

### 4. Run the continuous test sender

After the short test works, run the continuous NDI sender with the same fixed PM2.5 test value:

```bash
python td_ndi_bridge.py --device cpu --manual-pm25 25 --duration 20 --image-interval 8 --ndi-fps 2 --change-threshold 5 --poll-interval 60
```

Leave this Terminal process running while using TouchDesigner. Stop it with `Ctrl+C` when finished.

The NDI stream name is:

```text
Latent Mycelium NDI
```

## TouchDesigner

Open the TouchDesigner project:

```text
The-Latent-Mycelium-Live-Visual/The-Latent-Mycelium-Live-Visual.toe
```

In TouchDesigner:

1. Add or select an `NDI In TOP`.
2. Open its source/device menu.
3. Select `Latent Mycelium NDI`.

If you changed the sender name with `--ndi-name`, select that custom name instead.

## Which Environment To Use

This repository uses two Conda environments because the live NDI workflow and the ML notebook/training workflow have different dependency needs.

Use `ml-art-ndi` for:

- [td_ndi_bridge.py](td_ndi_bridge.py)
- NDI output to TouchDesigner
- live PM2.5/manual PM2.5 playback

Use `ml-art` for:

- Jupyter notebooks
- LoRA training
- Diffusers experiments outside the NDI sender
- MediaPipe and OpenCV work

Do not run `td_ndi_bridge.py` from `base` or `ml-art`. It should be run from `ml-art-ndi`.

## Install Environments

These instructions are tested on Apple Silicon macOS. Windows users may be able to run the project with Conda, CPU or CUDA PyTorch, TouchDesigner, and NDI installed, but the Windows setup is not fully tested.

If you only want to verify that the project runs on another computer, install `ml-art-ndi` first and skip `ml-art` until later.

Install `mamba` in the base Conda environment if needed:

```bash
conda install -n base -c conda-forge mamba
```

### Install `ml-art`

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

### Install `ml-art-ndi`

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

Expected output:

```text
buffered ndi env ok
```

## Live OpenAQ Data

The quick commands above use a fixed test value:

```bash
--manual-pm25 25
```

To use live OpenAQ data instead, remove `--manual-pm25`. The script automatically reads OpenAQ settings from the project `.env` file:

```text
OPENAQ_API_KEY=...
OPENAQ_LOCATION_ID=...
```

Create that `.env` file in the project root only if you want live API mode. It is not required for the fixed-value test commands in this README.

Run live API mode with:

```bash
conda activate ml-art-ndi
python td_ndi_bridge.py --device cpu --duration 20 --image-interval 8 --ndi-fps 2 --change-threshold 5 --poll-interval 60
```

This polls OpenAQ every 60 seconds. If `--manual-pm25` is included, the script uses the fixed test value and does not call the API.

## Common Parameters

- `--device cpu`: safest device option; slower but reliable.
- `--device mps`: uses Apple Metal if PyTorch MPS is available.
- `--device auto`: tries CUDA, then MPS, then CPU.
- `--manual-pm25 25`: uses a fixed PM2.5 value for testing.
- `--duration 20`: buffered clip length in seconds.
- `--image-interval 8`: seconds of clip time between generated keyframes.
- `--ndi-fps 2`: NDI playback frame rate.
- `--change-threshold 5`: only regenerate when PM2.5 changes enough.
- `--poll-interval 60`: poll live data every 60 seconds.
- `--run-once`: generate and play one buffered clip, then exit.

## Troubleshooting

### `KeyboardInterrupt` during `import torch`

This usually means `Ctrl+C` was pressed while Python was still loading PyTorch. Run the command again and wait.

CPU generation can be slow, especially the first time the model loads.

### `ModuleNotFoundError`

Check that the correct Conda environment is active:

```bash
conda activate ml-art-ndi
```

Then verify:

```bash
python -c "import NDIlib, torch; print('ok')"
```

### `CondaError: Run 'conda init' before 'conda activate'`

Do not run `/Users/yerie/miniforge3/condabin/conda activate ...` directly.

Use:

```bash
conda activate ml-art-ndi
```

If activation still fails in a new shell:

```bash
source ~/.zshrc
conda activate ml-art-ndi
```

### MPS does not work

Use CPU:

```bash
python td_ndi_bridge.py --device cpu --manual-pm25 25 --duration 8 --image-interval 4 --ndi-fps 2 --run-once
```

## Repository Layout

- [README.md](README.md): setup and running instructions.
- [environment.yml](environment.yml): maintained Conda environment for notebooks, training, and ML experiments.
- [ml-art-environment.yml](ml-art-environment.yml): exported snapshot of a known working environment.
- [train_text_to_image_lora.py](train_text_to_image_lora.py): LoRA training script.
- [td_ndi_bridge.py](td_ndi_bridge.py): buffered NDI renderer and sender for TouchDesigner playback.
- [dataset](dataset): training images, captions, and metadata.
- [training_runs](training_runs): trained LoRA outputs and previews.
- [The-Latent-Mycelium-Live-Visual](The-Latent-Mycelium-Live-Visual): TouchDesigner project files.

## VS Code And Jupyter

For notebooks and training work:

1. Open the project in VS Code.
2. Select the interpreter or kernel named `Python (ml-art)`.
3. Keep `ml-art-ndi` for terminal-based NDI runs.

## Author And Date

- Author: Yerie Ye
- Project date: 2026-04
- Repository: The Latent Mycelium

## Acknowledgements

This project builds on open-source ML, creative coding, and environmental data tools:

- Hugging Face Diffusers, Transformers, Accelerate, Safetensors, PEFT, and Datasets for Stable Diffusion LoRA training and inference.
- Stable Diffusion and latent diffusion research for the base text-to-image generation workflow.
- LoRA research by Hu et al. for efficient fine-tuning.
- PyTorch for model execution on CPU, MPS, or CUDA devices.
- OpenAQ for PM2.5 environmental data access.
- TouchDesigner for realtime visual playback and installation control.
- NDI and `ndi-python` for sending generated image buffers into TouchDesigner.
- PyFAD and course NDI examples for reference patterns used in the NDI sender workflow.

## Large Files

GitHub rejects regular Git files larger than 100MB.

Do not commit generated zip archives or other large exports. Zip files are ignored by this repository:

```text
*.zip
```

For large model artifacts, use Git LFS, Hugging Face, Google Drive, or GitHub Releases instead of committing them directly.
