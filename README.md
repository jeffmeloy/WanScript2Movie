# Script2Workflow

A Python tool that converts JSON movie scripts into ComfyUI workflows for automated video generation using the Wan 2.2 model.

## Features

- **Multi-turn video generation**: Process individual scenes or entire scripts
- **T2V and I2V support**: Text-to-video for first turn, image-to-video for subsequent turns
- **LoRA integration**: Automatic Lightning + optional LoRA loading
- **Memory management**: Built-in VRAM/RAM cleanup between turns
- **Flexible turn selection**: Process specific turns or ranges
- **Video chaining**: Automatically links turns using last frame as input image

## Requirements

- ComfyUI with the following nodes:
  - Wan 2.2 model files
  - GGUF loaders
  - VideoHelperSuite
  - rgthree Power Lora Loader
  - Memory cleanup nodes

## Usage

### Basic usage
```bash
python script2workflow.py movie_script.json
```

### Process specific turns
```bash
# Single turn
python script2workflow.py script.json --turns 3

# Range of turns  
python script2workflow.py script.json --turns 1:4
```

### Start with image (I2V for first turn)
```bash
python script2workflow.py script.json --image input_image.png
```

## Script Format

Your JSON script should follow this structure:
```json
{
  "turns": {
    "1": {
      "positive_prompt": "A beautiful sunset over mountains",
      "negative_prompt": "blurry, low quality"
    },
    "2": {
      "positive_prompt": "The camera pans to reveal a lake",
      "negative_prompt": "blurry, low quality"
    }
  }
}
```

## Output

- Individual turn videos: `{script_name}_turn{N}.mp4`
- Combined final video: `{script_name}_turns{range}_base_{timestamp}.mp4`
- ComfyUI workflow: `{script_name}_workflow.json`

## Configuration

Key parameters can be modified at the top of the script:
- Video resolution (default: 1280x720)
- Frame rate (default: 24fps)
- Video length (default: 81 frames)
- Model sampling shift (default: 8.0)
- Sampler settings (steps, CFG, scheduler)

## Model Files Required

- `wan_2.1_vae.safetensors`
- `umt5-xxl-encoder-Q5_K_S.gguf`
- Wan 2.2 T2V/I2V UNet models (high/low noise)
- Lightning LoRA files
- Optional style LoRAs
