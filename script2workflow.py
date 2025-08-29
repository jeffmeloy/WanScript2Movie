import json
import os
import sys
import argparse
import copy
import random
import uuid
from datetime import datetime

# --- CONSTANTS ---
HORIZONTAL_SPACING = 3600
NODE_ID_BASE_OFFSET = 1000

# Default model and file names
VAE_NAME = "wan_2.1_vae.safetensors"
CLIP_GGUF_NAME = "umt5-xxl-encoder-Q5_K_S.gguf"
T2V_HIGH_NOISE_UNET = "Wan2.2-T2V-A14B-HighNoise-Q8_0.gguf"
T2V_LOW_NOISE_UNET = "Wan2.2-T2V-A14B-LowNoise-Q8_0.gguf"

T2V_HIGH_NOISE_LORA = {
    "1": {
        "name": "Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_HIGH_fp16.safetensors",
        "enabled": True,
        "strength": 1.2,
    },
    "2": {
        "name": "Wan2.2 v3 - T2V - Insta Girls - HIGH 14B.safetensors",
        "enabled": False,
        "strength": 1.0,
    },
}

T2V_LOW_NOISE_LORA = {
    "1": {
        "name": "Wan2.2-Lightning_T2V-v1.1-A14B-4steps-lora_LOW_fp16.safetensors",
        "enabled": False,
        "strength": 1.0,
    },
    "2": {
        "name": "Wan2.2 v3 - T2V - Insta Girls - LOW - 14B.safetensors",
        "enabled": False,
        "strength": 1.0,
    },
}

I2V_HIGH_NOISE_UNET = "Wan2.2-I2V-A14B-HighNoise-Q8_0.gguf"
I2V_LOW_NOISE_UNET = "Wan2.2-I2V-A14B-LowNoise-Q8_0.gguf"

I2V_HIGH_NOISE_LORA = {
    "1": {
        "name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_HIGH_fp16.safetensors",
        "enabled": True,
        "strength": 1.5,
    },
    "2": {
        "name": "Wan2.2 v3 - T2V - Insta Girls - HIGH 14B.safetensors",
        "enabled": False,
        "strength": 0.5,
    }
}

I2V_LOW_NOISE_LORA = {
    "1": {
        "name": "Wan2.2-Lightning_I2V-A14B-4steps-lora_LOW_fp16.safetensors",
        "enabled": True,
        "strength": 1.1,
    },
    "2": {
        "name": "Wan2.2 v3 - T2V - Insta Girls - LOW - 14B.safetensors",
        "enabled": False,
        "strength": 0.5,
    }
}

# GLOBAL PARAMETERS
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
VIDEO_LENGTH = 81
VIDEO_BATCH_SIZE = 1
FRAME_RATE = 24

MODEL_SAMPLING_SHIFT = 8.0

FIRST_SAMPLER_NOISE_SEED = -1
NOISE_SEED_MODE = "randomize"
SAMPLER_STEPS_HIGH = 12
SAMPLER_STEPS_LOW = 12
SAMPLER_CFG = 1.0
SAMPLER_NAME = "euler"
SCHEDULER_NAME = "simple"

T2V_HIGH_START_STEP = 0
T2V_HIGH_END_STEP = 8
T2V_LOW_START_STEP = 8
T2V_LOW_END_STEP = 10000

I2V_HIGH_START_STEP = 0
I2V_HIGH_END_STEP = 8
I2V_LOW_START_STEP = 8
I2V_LOW_END_STEP = 10000

IMAGE_SELECT_INDEX = "-1"
IMAGE_SELECT_ERROR_FLAGS = [True, True]
BATCH_INPUT_COUNT = 2

VIDEO_FILENAME_PREFIX = "WAN2.2_movie"
VIDEO_FORMAT = "video/h264-mp4"
VIDEO_PIXEL_FORMAT = "yuv420p"
VIDEO_CRF = 19
VIDEO_LOOP_COUNT = 0
VIDEO_SAVE_METADATA = True
VIDEO_PINGPONG = False
VIDEO_SAVE_OUTPUT = True
VIDEO_TRIM_TO_AUDIO = False

# Image upscaling parameters
UPSCALE_METHOD = "lanczos"
UPSCALE_FACTOR = 2.0

NODE_TEMPLATES = {
    "VAELoader": {
        "type": "VAELoader",
        "size": [500.56, 58],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "name": "vae_name",
                "type": "COMBO",
                "widget": {"name": "vae_name"},
                "link": None,
            }
        ],
        "outputs": [
            {"name": "VAE", "type": "VAE", "links": [], "localized_name": "VAE"}
        ],
        "properties": {"Node name for S&R": "VAELoader"},
        "color": "#223",
        "bgcolor": "#335",
        "widgets_values": [],
    },
    "CLIPLoaderGGUF": {
        "type": "CLIPLoaderGGUF",
        "size": [497.75, 82],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "name": "clip_name",
                "type": "COMBO",
                "widget": {"name": "clip_name"},
                "link": None,
            },
            {"name": "type", "type": "COMBO", "widget": {"name": "type"}, "link": None},
        ],
        "outputs": [
            {"name": "CLIP", "type": "CLIP", "links": [], "localized_name": "CLIP"}
        ],
        "properties": {
            "cnr_id": "comfyui-gguf",
            "ver": "1.1.1",
            "Node name for S&R": "CLIPLoaderGGUF",
            "aux_id": "city96/ComfyUI-GGUF",
        },
        "color": "#223",
        "bgcolor": "#335",
        "widgets_values": [],
    },
    "UnetLoaderGGUF": {
        "type": "UnetLoaderGGUF",
        "size": [494.37, 58],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "name": "unet_name",
                "type": "COMBO",
                "widget": {"name": "unet_name"},
                "link": None,
            }
        ],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": [], "localized_name": "MODEL"}
        ],
        "properties": {
            "cnr_id": "ComfyUI-GGUF",
            "ver": "b3ec875a68d94b758914fd48d30571d953bb7a54",
            "Node name for S&R": "UnetLoaderGGUF",
            "aux_id": "city96/ComfyUI-GGUF",
        },
        "color": "#223",
        "bgcolor": "#335",
        "widgets_values": [],
    },
    "ModelSamplingSD3": {
        "type": "ModelSamplingSD3",
        "size": [448.5, 58],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {
                "name": "shift",
                "type": "FLOAT",
                "widget": {"name": "shift"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": [], "localized_name": "MODEL"}
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.46",
            "Node name for S&R": "ModelSamplingSD3",
        },
        "widgets_values": [],
    },
    "Power Lora Loader (rgthree)": {
        "type": "Power Lora Loader (rgthree)",
        "size": [514.27, 217.6],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "clip", "type": "CLIP", "link": None},
        ],
        "outputs": [
            {"name": "MODEL", "type": "MODEL", "links": []},
            {"name": "CLIP", "type": "CLIP", "links": []},
        ],
        "properties": {
            "cnr_id": "rgthree-comfy",
            "ver": "1.0.2507112302",
            "Show Strengths": "Single Strength",
            "aux_id": "rgthree/rgthree-comfy",
        },
        "color": "#323",
        "bgcolor": "#535",
        "widgets_values": [],
    },
    "CLIPTextEncode": {
        "type": "CLIPTextEncode",
        "size": [508.7, 118.61],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "clip", "type": "CLIP", "link": None},
            {
                "name": "text",
                "type": "STRING",
                "widget": {"name": "text"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "CONDITIONING",
                "type": "CONDITIONING",
                "links": [],
                "localized_name": "CONDITIONING",
            }
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.43",
            "Node name for S&R": "CLIPTextEncode",
        },
        "widgets_values": [],
    },
    "EmptyHunyuanLatentVideo": {
        "type": "EmptyHunyuanLatentVideo",
        "size": [499.97, 134.53],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {
                "name": "height",
                "type": "INT",
                "widget": {"name": "height"},
                "link": None,
            },
            {
                "name": "length",
                "type": "INT",
                "widget": {"name": "length"},
                "link": None,
            },
            {
                "name": "batch_size",
                "type": "INT",
                "widget": {"name": "batch_size"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "LATENT",
                "type": "LATENT",
                "links": [],
                "localized_name": "LATENT",
            }
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.43",
            "Node name for S&R": "EmptyHunyuanLatentVideo",
        },
        "widgets_values": [],
    },
    "KSamplerAdvanced": {
        "type": "KSamplerAdvanced",
        "size": [244.74, 334],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "model", "type": "MODEL", "link": None},
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "latent_image", "type": "LATENT", "link": None},
            {
                "name": "add_noise",
                "type": "COMBO",
                "widget": {"name": "add_noise"},
                "link": None,
            },
            {
                "name": "noise_seed",
                "type": "INT",
                "widget": {"name": "noise_seed"},
                "link": None,
            },
            {"name": "steps", "type": "INT", "widget": {"name": "steps"}, "link": None},
            {"name": "cfg", "type": "FLOAT", "widget": {"name": "cfg"}, "link": None},
            {
                "name": "sampler_name",
                "type": "COMBO",
                "widget": {"name": "sampler_name"},
                "link": None,
            },
            {
                "name": "scheduler",
                "type": "COMBO",
                "widget": {"name": "scheduler"},
                "link": None,
            },
            {
                "name": "start_at_step",
                "type": "INT",
                "widget": {"name": "start_at_step"},
                "link": None,
            },
            {
                "name": "end_at_step",
                "type": "INT",
                "widget": {"name": "end_at_step"},
                "link": None,
            },
            {
                "name": "return_with_leftover_noise",
                "type": "COMBO",
                "widget": {"name": "return_with_leftover_noise"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "LATENT",
                "type": "LATENT",
                "links": [],
                "localized_name": "LATENT",
            }
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.46",
            "Node name for S&R": "KSamplerAdvanced",
        },
        "widgets_values": [],
    },
    "VRAMCleanup": {
        "type": "VRAMCleanup",
        "size": [270, 82],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": "anything", "type": "*", "link": None}],
        "outputs": [
            {"name": "output", "type": "*", "links": [], "localized_name": "output"}
        ],
        "properties": {
            "cnr_id": "comfyui_memory_cleanup",
            "ver": "1.1.1",
            "Node name for S&R": "VRAMCleanup",
        },
        "widgets_values": [True, True],
    },
    "RAMCleanup": {
        "type": "RAMCleanup",
        "size": [270, 130],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": "anything", "type": "*", "link": None}],
        "outputs": [
            {"name": "output", "type": "*", "links": [], "localized_name": "output"}
        ],
        "properties": {
            "cnr_id": "comfyui_memory_cleanup",
            "ver": "1.1.1",
            "Node name for S&R": "RAMCleanup",
        },
        "widgets_values": [True, True, True, 1],
    },
    "VAEDecode": {
        "type": "VAEDecode",
        "size": [155.52, 46],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "samples", "type": "LATENT", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "localized_name": "IMAGE"}
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.43",
            "Node name for S&R": "VAEDecode",
        },
        "color": "#323",
        "bgcolor": "#535",
        "widgets_values": [],
    },
    "VHS_SelectImages": {
        "type": "VHS_SelectImages",
        "size": [210, 106],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": None},
            {
                "name": "indexes",
                "type": "STRING",
                "widget": {"name": "indexes"},
                "link": None,
            },
            {
                "name": "err_if_missing",
                "type": "BOOLEAN",
                "widget": {"name": "err_if_missing"},
                "link": None,
            },
            {
                "name": "err_if_empty",
                "type": "BOOLEAN",
                "widget": {"name": "err_if_empty"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "localized_name": "IMAGE"}
        ],
        "properties": {
            "cnr_id": "comfyui-videohelpersuite",
            "ver": "8e4d79471bf1952154768e8435a9300077b534fa",
            "Node name for S&R": "VHS_SelectImages",
        },
        "color": "#233",
        "bgcolor": "#355",
        "widgets_values": [],
    },
    "WanImageToVideo": {
        "type": "WanImageToVideo",
        "size": [492.73, 210],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "positive", "type": "CONDITIONING", "link": None},
            {"name": "negative", "type": "CONDITIONING", "link": None},
            {"name": "vae", "type": "VAE", "link": None},
            {
                "name": "clip_vision_output",
                "type": "CLIP_VISION_OUTPUT",
                "shape": 7,
                "link": None,
            },
            {"name": "start_image", "type": "IMAGE", "shape": 7, "link": None},
            {"name": "width", "type": "INT", "widget": {"name": "width"}, "link": None},
            {
                "name": "height",
                "type": "INT",
                "widget": {"name": "height"},
                "link": None,
            },
            {
                "name": "length",
                "type": "INT",
                "widget": {"name": "length"},
                "link": None,
            },
            {
                "name": "batch_size",
                "type": "INT",
                "widget": {"name": "batch_size"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "positive",
                "type": "CONDITIONING",
                "links": [],
                "localized_name": "positive",
            },
            {
                "name": "negative",
                "type": "CONDITIONING",
                "links": [],
                "localized_name": "negative",
            },
            {
                "name": "latent",
                "type": "LATENT",
                "links": [],
                "localized_name": "latent",
            },
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.47",
            "Node name for S&R": "WanImageToVideo",
        },
        "widgets_values": [],
    },
    "ImageBatchMulti": {
        "type": "ImageBatchMulti",
        "size": [270, 102],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "image_1", "type": "IMAGE", "link": None},
            {"name": "image_2", "type": "IMAGE", "shape": 7, "link": None},
            {
                "name": "inputcount",
                "type": "INT",
                "widget": {"name": "inputcount"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "images", "type": "IMAGE", "links": [], "localized_name": "images"}
        ],
        "properties": {
            "cnr_id": "comfyui-kjnodes",
            "ver": "2f7300dc546ec2d36fa8b0feebe493d41026524c",
            "Node name for S&R": "ImageBatchMulti",
        },
        "color": "#233",
        "bgcolor": "#355",
        "widgets_values": [],
    },
    "VHS_VideoCombine": {
        "type": "VHS_VideoCombine",
        "size": [659.25, 334],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            {"name": "audio", "type": "AUDIO", "shape": 7, "link": None},
            {
                "name": "meta_batch",
                "type": "VHS_BatchManager",
                "shape": 7,
                "link": None,
            },
            {"name": "vae", "type": "VAE", "shape": 7, "link": None},
            {
                "name": "frame_rate",
                "type": "FLOAT",
                "widget": {"name": "frame_rate"},
                "link": None,
            },
            {
                "name": "loop_count",
                "type": "INT",
                "widget": {"name": "loop_count"},
                "link": None,
            },
            {
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
                "link": None,
            },
            {
                "name": "format",
                "type": "COMBO",
                "widget": {"name": "format"},
                "link": None,
            },
            {
                "name": "pingpong",
                "type": "BOOLEAN",
                "widget": {"name": "pingpong"},
                "link": None,
            },
            {
                "name": "save_output",
                "type": "BOOLEAN",
                "widget": {"name": "save_output"},
                "link": None,
            },
            {
                "name": "pix_fmt",
                "type": "COMBO",
                "widget": {"name": "pix_fmt"},
                "link": None,
            },
            {"name": "crf", "type": "INT", "widget": {"name": "crf"}, "link": None},
            {
                "name": "save_metadata",
                "type": "BOOLEAN",
                "widget": {"name": "save_metadata"},
                "link": None,
            },
            {
                "name": "trim_to_audio",
                "type": "BOOLEAN",
                "widget": {"name": "trim_to_audio"},
                "link": None,
            },
        ],
        "outputs": [
            {
                "name": "Filenames",
                "type": "VHS_FILENAMES",
                "links": [],
                "localized_name": "Filenames",
            }
        ],
        "properties": {
            "cnr_id": "comfyui-videohelpersuite",
            "ver": "1.7.2",
            "Node name for S&R": "VHS_VideoCombine",
            "aux_id": "Kosinkadink/ComfyUI-VideoHelperSuite",
        },
        "widgets_values": [],
    },
    "VHS_LoadVideo": {
        "type": "VHS_LoadVideo",
        "size": [247.46, 310],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "name": "meta_batch",
                "type": "VHS_BatchManager",
                "shape": 7,
                "link": None,
            },
            {"name": "vae", "type": "VAE", "shape": 7, "link": None},
            {
                "name": "video",
                "type": "COMBO",
                "widget": {"name": "video"},
                "link": None,
            },
            {
                "name": "force_rate",
                "type": "FLOAT",
                "widget": {"name": "force_rate"},
                "link": None,
            },
            {
                "name": "custom_width",
                "type": "INT",
                "widget": {"name": "custom_width"},
                "link": None,
            },
            {
                "name": "custom_height",
                "type": "INT",
                "widget": {"name": "custom_height"},
                "link": None,
            },
            {
                "name": "frame_load_cap",
                "type": "INT",
                "widget": {"name": "frame_load_cap"},
                "link": None,
            },
            {
                "name": "skip_first_frames",
                "type": "INT",
                "widget": {"name": "skip_first_frames"},
                "link": None,
            },
            {
                "name": "select_every_nth",
                "type": "INT",
                "widget": {"name": "select_every_nth"},
                "link": None,
            },
            {
                "name": "format",
                "type": "COMBO",
                "widget": {"name": "format"},
                "shape": 7,
                "link": None,
            },
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "localized_name": "IMAGE"},
            {
                "name": "frame_count",
                "type": "INT",
                "links": [],
                "localized_name": "frame_count",
            },
            {"name": "audio", "type": "AUDIO", "links": [], "localized_name": "audio"},
            {
                "name": "video_info",
                "type": "VHS_VIDEOINFO",
                "links": [],
                "localized_name": "video_info",
            },
        ],
        "properties": {
            "cnr_id": "comfyui-videohelpersuite",
            "ver": "4c7858ddd5126f7293dc3c9f6e0fc4c263cde079",
            "Node name for S&R": "VHS_LoadVideo",
            "aux_id": "Kosinkadink/ComfyUI-VideoHelperSuite",
        },
        "widgets_values": [],
    },
    "VHS_VideoInfo": {
        "type": "VHS_VideoInfo",
        "size": [225.6, 206],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [{"name": "video_info", "type": "VHS_VIDEOINFO", "link": None}],
        "outputs": [
            {
                "name": "source_fpsÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
                "type": "FLOAT",
                "links": [],
                "localized_name": "source_fpsÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
            },
            {
                "name": "source_frame_countÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
                "type": "INT",
                "links": [],
                "localized_name": "source_frame_countÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
            },
            {
                "name": "source_durationÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
                "type": "FLOAT",
                "links": [],
                "localized_name": "source_durationÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
            },
            {
                "name": "source_widthÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
                "type": "INT",
                "links": [],
                "localized_name": "source_widthÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
            },
            {
                "name": "source_heightÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
                "type": "INT",
                "links": [],
                "localized_name": "source_heightÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¨",
            },
            {
                "name": "loaded_fpsÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
                "type": "FLOAT",
                "links": [],
                "localized_name": "loaded_fpsÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
            },
            {
                "name": "loaded_frame_countÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
                "type": "INT",
                "links": [],
                "localized_name": "loaded_frame_countÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
            },
            {
                "name": "loaded_durationÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
                "type": "FLOAT",
                "links": [],
                "localized_name": "loaded_durationÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
            },
            {
                "name": "loaded_widthÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
                "type": "INT",
                "links": [],
                "localized_name": "loaded_widthÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
            },
            {
                "name": "loaded_heightÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
                "type": "INT",
                "links": [],
                "localized_name": "loaded_heightÃƒÂ°Ã…Â¸Ã…Â¸Ã‚Â¦",
            },
        ],
        "properties": {
            "cnr_id": "comfyui-videohelpersuite",
            "ver": "1.7.2",
            "Node name for S&R": "VHS_VideoInfo",
            "aux_id": "Kosinkadink/ComfyUI-VideoHelperSuite",
        },
        "widgets_values": [],
    },
    "LoadImage": {
        "type": "LoadImage",
        "size": [315, 314],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {
                "name": "image",
                "type": "COMBO",
                "widget": {"name": "image"},
                "link": None,
            }
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "localized_name": "IMAGE"},
            {"name": "MASK", "type": "MASK", "links": [], "localized_name": "MASK"},
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.43",
            "Node name for S&R": "LoadImage",
        },
        "color": "#323",
        "bgcolor": "#535",
        "widgets_values": [],
    },
    "SaveImage": {
        "type": "SaveImage",
        "size": [315, 58],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "images", "type": "IMAGE", "link": None},
            {
                "name": "filename_prefix",
                "type": "STRING",
                "widget": {"name": "filename_prefix"},
                "link": None,
            },
        ],
        "outputs": [],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.43",
            "Node name for S&R": "SaveImage",
        },
        "color": "#323",
        "bgcolor": "#535",
        "widgets_values": [],
    },
    "ImageScaleBy": {
        "type": "ImageScaleBy",
        "size": [270, 82],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            {"name": "image", "type": "IMAGE", "link": None},
            {
                "name": "upscale_method",
                "type": "COMBO",
                "widget": {"name": "upscale_method"},
                "link": None,
            },
            {
                "name": "scale_by",
                "type": "FLOAT",
                "widget": {"name": "scale_by"},
                "link": None,
            },
        ],
        "outputs": [
            {"name": "IMAGE", "type": "IMAGE", "links": [], "localized_name": "IMAGE"}
        ],
        "properties": {
            "cnr_id": "comfy-core",
            "ver": "0.3.50",
            "Node name for S&R": "ImageScaleBy",
        },
        "widgets_values": [],
    },
}


def parse_turns_range(turns_str):
    if ":" in turns_str:
        start, end = turns_str.split(":", 1)
        return list(range(int(start), int(end) + 1))
    else:
        return [int(turns_str)]


def create_multi_lora_config(lora_dict):
    widgets_values = [{}, {"type": "PowerLoraLoaderHeaderWidget"}]
    for lora_config in lora_dict.values():
        if lora_config.get("enabled", True):
            widgets_values.append(
                {
                    "on": True,
                    "lora": lora_config["name"],
                    "strength": lora_config["strength"],
                    "strengthTwo": None,
                }
            )
    widgets_values.extend([{}, ""])
    return widgets_values


def create_node(node_type, node_id, pos, widgets_override=None, title=None):
    if node_type not in NODE_TEMPLATES:
        raise ValueError(f"Node type '{node_type}' not found in templates.")

    node = copy.deepcopy(NODE_TEMPLATES[node_type])
    node["id"] = node_id
    node["pos"] = pos

    if widgets_override is not None:
        node["widgets_values"] = widgets_override

    if "outputs" in node:
        for output in node["outputs"]:
            if "links" in output and output["links"] is not None:
                output["links"] = []

    if title is not None:
        node["title"] = title

    if node_type == "CLIPTextEncode":
        if title and "Positive" in title:
            node["color"] = "#232"
            node["bgcolor"] = "#353"
        elif title and "Negative" in title:
            node["color"] = "#322"
            node["bgcolor"] = "#533"

    return node


def create_first_turn_i2v(
    turn_idx, positive_prompt, negative_prompt, image_filename, workflow_name
):
    nodes = []
    base_id = (turn_idx - 1) * NODE_ID_BASE_OFFSET
    x_pos = (turn_idx - 1) * HORIZONTAL_SPACING

    if FIRST_SAMPLER_NOISE_SEED == -1:
        noise_seed_value = random.randint(0, 2**32 - 1)
    else:
        noise_seed_value = FIRST_SAMPLER_NOISE_SEED

    ids = {
        "vae_loader": base_id + 1,
        "clip_loader": base_id + 2,
        "unet_high": base_id + 3,
        "unet_low": base_id + 4,
        "sampler_high": base_id + 5,
        "sampler_low": base_id + 6,
        "lora_high": base_id + 7,
        "lora_low": base_id + 8,
        "prompt_pos": base_id + 9,
        "prompt_neg": base_id + 10,
        "i2v_latent": base_id + 11,
        "ksampler_high": base_id + 12,
        "ksampler_low": base_id + 13,
        "vram_cleanup_mid": base_id + 14,
        "vae_decode": base_id + 15,
        "vram_cleanup_final": base_id + 16,
        "ram_cleanup_final": base_id + 17,
        "load_image": base_id + 18,
        "ram_cleanup_mid": base_id + 19,
        "turn_video": base_id + 20,
        "image_scale": base_id + 21,
    }

    high_lora_config = create_multi_lora_config(I2V_HIGH_NOISE_LORA)
    low_lora_config = create_multi_lora_config(I2V_LOW_NOISE_LORA)

    nodes.extend(
        [
            create_node("VAELoader", ids["vae_loader"], [x_pos, 300], [VAE_NAME]),
            create_node(
                "CLIPLoaderGGUF",
                ids["clip_loader"],
                [x_pos, 400],
                [CLIP_GGUF_NAME, "wan"],
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_high"], [x_pos, 100], [I2V_HIGH_NOISE_UNET]
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_low"], [x_pos, 200], [I2V_LOW_NOISE_UNET]
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_high"],
                [x_pos + 550, 100],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_low"],
                [x_pos + 550, 200],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_high"],
                [x_pos, 500],
                high_lora_config,
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_low"],
                [x_pos, 650],
                low_lora_config,
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_pos"],
                [x_pos, 900],
                [positive_prompt],
                "Positive Prompt",
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_neg"],
                [x_pos, 1050],
                [negative_prompt],
                "Negative Prompt",
            ),
            create_node(
                "LoadImage", ids["load_image"], [x_pos - 400, 1200], [image_filename]
            ),
            create_node(
                "ImageScaleBy",
                ids["image_scale"],
                [x_pos - 200, 1200],
                [UPSCALE_METHOD, UPSCALE_FACTOR],
            ),
            create_node(
                "WanImageToVideo",
                ids["i2v_latent"],
                [x_pos, 1200],
                [VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_LENGTH, VIDEO_BATCH_SIZE],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_high"],
                [x_pos + 900, 300],
                [
                    "enable",
                    noise_seed_value,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_HIGH,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    I2V_HIGH_START_STEP,
                    I2V_HIGH_END_STEP,
                    "enable",
                ],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_low"],
                [x_pos + 1200, 300],
                [
                    "disable",
                    1,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_LOW,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    I2V_LOW_START_STEP,
                    I2V_LOW_END_STEP,
                    "disable",
                ],
            ),
            create_node("VRAMCleanup", ids["vram_cleanup_mid"], [x_pos + 900, 700], []),
            create_node("RAMCleanup", ids["ram_cleanup_mid"], [x_pos + 1050, 700], []),
            create_node("VAEDecode", ids["vae_decode"], [x_pos + 1500, 300], []),
            create_node(
                "VRAMCleanup", ids["vram_cleanup_final"], [x_pos + 1500, 400], []
            ),
            create_node(
                "RAMCleanup", ids["ram_cleanup_final"], [x_pos + 1500, 500], []
            ),
            create_node(
                "VHS_VideoCombine",
                ids["turn_video"],
                [x_pos + 1800, 600],
                [
                    FRAME_RATE,
                    VIDEO_LOOP_COUNT,
                    f"{workflow_name}_turn{turn_idx}",
                    VIDEO_FORMAT,
                    VIDEO_PINGPONG,
                    VIDEO_SAVE_OUTPUT,
                    VIDEO_PIXEL_FORMAT,
                    VIDEO_CRF,
                    VIDEO_SAVE_METADATA,
                    VIDEO_TRIM_TO_AUDIO,
                ],
                f"Save Turn {turn_idx} Video",
            ),
        ]
    )

    link_defs = [
        (ids["unet_high"], 0, ids["sampler_high"], 0, "MODEL"),
        (ids["unet_low"], 0, ids["sampler_low"], 0, "MODEL"),
        (ids["sampler_high"], 0, ids["lora_high"], 0, "MODEL"),
        (ids["sampler_low"], 0, ids["lora_low"], 0, "MODEL"),
        (ids["clip_loader"], 0, ids["lora_high"], 1, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_pos"], 0, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_neg"], 0, "CLIP"),
        (ids["lora_high"], 0, ids["ksampler_high"], 0, "MODEL"),
        (ids["lora_low"], 0, ids["ksampler_low"], 0, "MODEL"),
        (ids["prompt_pos"], 0, ids["i2v_latent"], 0, "CONDITIONING"),
        (ids["prompt_neg"], 0, ids["i2v_latent"], 1, "CONDITIONING"),
        (ids["vae_loader"], 0, ids["i2v_latent"], 2, "VAE"),
        (ids["load_image"], 0, ids["image_scale"], 0, "IMAGE"),
        (ids["image_scale"], 0, ids["i2v_latent"], 4, "IMAGE"),
        (ids["i2v_latent"], 0, ids["ksampler_high"], 1, "CONDITIONING"),
        (ids["i2v_latent"], 0, ids["ksampler_low"], 1, "CONDITIONING"),
        (ids["i2v_latent"], 1, ids["ksampler_high"], 2, "CONDITIONING"),
        (ids["i2v_latent"], 1, ids["ksampler_low"], 2, "CONDITIONING"),
        (ids["i2v_latent"], 2, ids["ksampler_high"], 3, "LATENT"),
        (ids["ksampler_high"], 0, ids["vram_cleanup_mid"], 0, "*"),
        (ids["vram_cleanup_mid"], 0, ids["ram_cleanup_mid"], 0, "*"),
        (ids["ram_cleanup_mid"], 0, ids["ksampler_low"], 3, "LATENT"),
        (ids["ksampler_low"], 0, ids["vae_decode"], 0, "LATENT"),
        (ids["vae_loader"], 0, ids["vae_decode"], 1, "VAE"),
        (ids["vae_decode"], 0, ids["vram_cleanup_final"], 0, "*"),
        (ids["vram_cleanup_final"], 0, ids["ram_cleanup_final"], 0, "*"),
        (ids["vae_decode"], 0, ids["turn_video"], 0, "IMAGE"),
    ]

    return nodes, link_defs, ids["ram_cleanup_final"]


def create_t2v_turn(turn_idx, positive_prompt, negative_prompt, workflow_name):
    nodes = []
    base_id = (turn_idx - 1) * NODE_ID_BASE_OFFSET
    x_pos = (turn_idx - 1) * HORIZONTAL_SPACING

    if FIRST_SAMPLER_NOISE_SEED == -1:
        noise_seed_value = random.randint(0, 2**32 - 1)
    else:
        noise_seed_value = FIRST_SAMPLER_NOISE_SEED

    ids = {
        "vae_loader": base_id + 1,
        "clip_loader": base_id + 2,
        "unet_high": base_id + 3,
        "unet_low": base_id + 4,
        "sampler_high": base_id + 5,
        "sampler_low": base_id + 6,
        "lora_high": base_id + 7,
        "lora_low": base_id + 8,
        "prompt_pos": base_id + 9,
        "prompt_neg": base_id + 10,
        "empty_latent": base_id + 11,
        "ksampler_high": base_id + 12,
        "ksampler_low": base_id + 13,
        "vram_cleanup_mid": base_id + 14,
        "vae_decode": base_id + 15,
        "vram_cleanup_final": base_id + 16,
        "ram_cleanup_final": base_id + 17,
        "ram_cleanup_mid": base_id + 18,
        "turn_video": base_id + 19,
    }

    high_lora_config = create_multi_lora_config(T2V_HIGH_NOISE_LORA)
    low_lora_config = create_multi_lora_config(T2V_LOW_NOISE_LORA)

    nodes.extend(
        [
            create_node("VAELoader", ids["vae_loader"], [x_pos, 300], [VAE_NAME]),
            create_node(
                "CLIPLoaderGGUF",
                ids["clip_loader"],
                [x_pos, 400],
                [CLIP_GGUF_NAME, "wan"],
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_high"], [x_pos, 100], [T2V_HIGH_NOISE_UNET]
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_low"], [x_pos, 200], [T2V_LOW_NOISE_UNET]
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_high"],
                [x_pos + 550, 100],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_low"],
                [x_pos + 550, 200],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_high"],
                [x_pos, 500],
                high_lora_config,
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_low"],
                [x_pos, 650],
                low_lora_config,
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_pos"],
                [x_pos, 900],
                [positive_prompt],
                "Positive Prompt",
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_neg"],
                [x_pos, 1050],
                [negative_prompt],
                "Negative Prompt",
            ),
            create_node(
                "EmptyHunyuanLatentVideo",
                ids["empty_latent"],
                [x_pos, 1200],
                [VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_LENGTH, VIDEO_BATCH_SIZE],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_high"],
                [x_pos + 900, 300],
                [
                    "enable",
                    noise_seed_value,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_HIGH,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    T2V_HIGH_START_STEP,
                    T2V_HIGH_END_STEP,
                    "enable",
                ],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_low"],
                [x_pos + 1200, 300],
                [
                    "disable",
                    1,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_LOW,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    T2V_LOW_START_STEP,
                    T2V_LOW_END_STEP,
                    "disable",
                ],
            ),
            create_node("VRAMCleanup", ids["vram_cleanup_mid"], [x_pos + 900, 700], []),
            create_node("RAMCleanup", ids["ram_cleanup_mid"], [x_pos + 1050, 700], []),
            create_node("VAEDecode", ids["vae_decode"], [x_pos + 1500, 300], []),
            create_node(
                "VRAMCleanup", ids["vram_cleanup_final"], [x_pos + 1500, 400], []
            ),
            create_node(
                "RAMCleanup", ids["ram_cleanup_final"], [x_pos + 1500, 500], []
            ),
            create_node(
                "VHS_VideoCombine",
                ids["turn_video"],
                [x_pos + 1800, 600],
                [
                    FRAME_RATE,
                    VIDEO_LOOP_COUNT,
                    f"{workflow_name}_turn{turn_idx}",
                    VIDEO_FORMAT,
                    VIDEO_PINGPONG,
                    VIDEO_SAVE_OUTPUT,
                    VIDEO_PIXEL_FORMAT,
                    VIDEO_CRF,
                    VIDEO_SAVE_METADATA,
                    VIDEO_TRIM_TO_AUDIO,
                ],
                f"Save Turn {turn_idx} Video",
            ),
        ]
    )

    link_defs = [
        (ids["unet_high"], 0, ids["sampler_high"], 0, "MODEL"),
        (ids["unet_low"], 0, ids["sampler_low"], 0, "MODEL"),
        (ids["sampler_high"], 0, ids["lora_high"], 0, "MODEL"),
        (ids["sampler_low"], 0, ids["lora_low"], 0, "MODEL"),
        (ids["clip_loader"], 0, ids["lora_high"], 1, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_pos"], 0, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_neg"], 0, "CLIP"),
        (ids["lora_high"], 0, ids["ksampler_high"], 0, "MODEL"),
        (ids["lora_low"], 0, ids["ksampler_low"], 0, "MODEL"),
        (ids["prompt_pos"], 0, ids["ksampler_high"], 1, "CONDITIONING"),
        (ids["prompt_pos"], 0, ids["ksampler_low"], 1, "CONDITIONING"),
        (ids["prompt_neg"], 0, ids["ksampler_high"], 2, "CONDITIONING"),
        (ids["prompt_neg"], 0, ids["ksampler_low"], 2, "CONDITIONING"),
        (ids["empty_latent"], 0, ids["ksampler_high"], 3, "LATENT"),
        (ids["ksampler_high"], 0, ids["vram_cleanup_mid"], 0, "*"),
        (ids["vram_cleanup_mid"], 0, ids["ram_cleanup_mid"], 0, "*"),
        (ids["ram_cleanup_mid"], 0, ids["ksampler_low"], 3, "LATENT"),
        (ids["ksampler_low"], 0, ids["vae_decode"], 0, "LATENT"),
        (ids["vae_loader"], 0, ids["vae_decode"], 1, "VAE"),
        (ids["vae_decode"], 0, ids["vram_cleanup_final"], 0, "*"),
        (ids["vram_cleanup_final"], 0, ids["ram_cleanup_final"], 0, "*"),
        (ids["vae_decode"], 0, ids["turn_video"], 0, "IMAGE"),
    ]

    return nodes, link_defs, ids["ram_cleanup_final"]


def create_i2v_turn(
    turn_idx, positive_prompt, negative_prompt, prev_turn_output_node_id, workflow_name
):
    nodes = []
    base_id = (turn_idx - 1) * NODE_ID_BASE_OFFSET
    x_pos = (turn_idx - 1) * HORIZONTAL_SPACING

    if FIRST_SAMPLER_NOISE_SEED == -1:
        noise_seed_value = random.randint(0, 2**32 - 1)
    else:
        noise_seed_value = FIRST_SAMPLER_NOISE_SEED

    ids = {
        "vae_loader": base_id + 1,
        "clip_loader": base_id + 2,
        "unet_high": base_id + 3,
        "unet_low": base_id + 4,
        "sampler_high": base_id + 5,
        "sampler_low": base_id + 6,
        "lora_high": base_id + 7,
        "lora_low": base_id + 8,
        "prompt_pos": base_id + 9,
        "prompt_neg": base_id + 10,
        "i2v_latent": base_id + 11,
        "ksampler_high": base_id + 12,
        "ksampler_low": base_id + 13,
        "vram_cleanup_mid": base_id + 14,
        "vae_decode": base_id + 15,
        "vram_cleanup_final": base_id + 16,
        "ram_cleanup_final": base_id + 17,
        "select_image": base_id + 18,
        "batch_images": base_id + 19,
        "ram_cleanup_mid": base_id + 20,
        "turn_video": base_id + 21,
        "image_scale": base_id + 22,
    }

    high_lora_config = create_multi_lora_config(I2V_HIGH_NOISE_LORA)
    low_lora_config = create_multi_lora_config(I2V_LOW_NOISE_LORA)

    nodes.extend(
        [
            create_node("VAELoader", ids["vae_loader"], [x_pos, 300], [VAE_NAME]),
            create_node(
                "CLIPLoaderGGUF",
                ids["clip_loader"],
                [x_pos, 400],
                [CLIP_GGUF_NAME, "wan"],
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_high"], [x_pos, 100], [I2V_HIGH_NOISE_UNET]
            ),
            create_node(
                "UnetLoaderGGUF", ids["unet_low"], [x_pos, 200], [I2V_LOW_NOISE_UNET]
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_high"],
                [x_pos + 550, 100],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "ModelSamplingSD3",
                ids["sampler_low"],
                [x_pos + 550, 200],
                [MODEL_SAMPLING_SHIFT],
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_high"],
                [x_pos, 500],
                high_lora_config,
            ),
            create_node(
                "Power Lora Loader (rgthree)",
                ids["lora_low"],
                [x_pos, 650],
                low_lora_config,
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_pos"],
                [x_pos, 900],
                [positive_prompt],
                "Positive Prompt",
            ),
            create_node(
                "CLIPTextEncode",
                ids["prompt_neg"],
                [x_pos, 1050],
                [negative_prompt],
                "Negative Prompt",
            ),
            create_node(
                "VHS_SelectImages",
                ids["select_image"],
                [x_pos - 400, 1200],
                [IMAGE_SELECT_INDEX] + IMAGE_SELECT_ERROR_FLAGS,
            ),
            create_node(
                "ImageScaleBy",
                ids["image_scale"],
                [x_pos - 200, 1200],
                [UPSCALE_METHOD, UPSCALE_FACTOR],
            ),
            create_node(
                "WanImageToVideo",
                ids["i2v_latent"],
                [x_pos, 1200],
                [VIDEO_WIDTH, VIDEO_HEIGHT, VIDEO_LENGTH, VIDEO_BATCH_SIZE],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_high"],
                [x_pos + 900, 300],
                [
                    "enable",
                    noise_seed_value,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_HIGH,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    I2V_HIGH_START_STEP,
                    I2V_HIGH_END_STEP,
                    "enable",
                ],
            ),
            create_node(
                "KSamplerAdvanced",
                ids["ksampler_low"],
                [x_pos + 1200, 300],
                [
                    "disable",
                    1,
                    NOISE_SEED_MODE,
                    SAMPLER_STEPS_LOW,
                    SAMPLER_CFG,
                    SAMPLER_NAME,
                    SCHEDULER_NAME,
                    I2V_LOW_START_STEP,
                    I2V_LOW_END_STEP,
                    "disable",
                ],
            ),
            create_node("VRAMCleanup", ids["vram_cleanup_mid"], [x_pos + 900, 700], []),
            create_node("RAMCleanup", ids["ram_cleanup_mid"], [x_pos + 1050, 700], []),
            create_node("VAEDecode", ids["vae_decode"], [x_pos + 1500, 300], []),
            create_node(
                "VRAMCleanup", ids["vram_cleanup_final"], [x_pos + 1500, 400], []
            ),
            create_node(
                "RAMCleanup", ids["ram_cleanup_final"], [x_pos + 1500, 500], []
            ),
            create_node(
                "ImageBatchMulti",
                ids["batch_images"],
                [x_pos + 1800, 300],
                [BATCH_INPUT_COUNT, None],
            ),
            create_node(
                "VHS_VideoCombine",
                ids["turn_video"],
                [x_pos + 1500, 700],
                [
                    FRAME_RATE,
                    VIDEO_LOOP_COUNT,
                    f"{workflow_name}_turn{turn_idx}",
                    VIDEO_FORMAT,
                    VIDEO_PINGPONG,
                    VIDEO_SAVE_OUTPUT,
                    VIDEO_PIXEL_FORMAT,
                    VIDEO_CRF,
                    VIDEO_SAVE_METADATA,
                    VIDEO_TRIM_TO_AUDIO,
                ],
                f"Save Turn {turn_idx} Video",
            ),
        ]
    )

    link_defs = [
        (ids["unet_high"], 0, ids["sampler_high"], 0, "MODEL"),
        (ids["unet_low"], 0, ids["sampler_low"], 0, "MODEL"),
        (ids["sampler_high"], 0, ids["lora_high"], 0, "MODEL"),
        (ids["sampler_low"], 0, ids["lora_low"], 0, "MODEL"),
        (ids["clip_loader"], 0, ids["lora_high"], 1, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_pos"], 0, "CLIP"),
        (ids["lora_high"], 1, ids["prompt_neg"], 0, "CLIP"),
        (ids["lora_high"], 0, ids["ksampler_high"], 0, "MODEL"),
        (ids["lora_low"], 0, ids["ksampler_low"], 0, "MODEL"),
        (ids["prompt_pos"], 0, ids["i2v_latent"], 0, "CONDITIONING"),
        (ids["prompt_neg"], 0, ids["i2v_latent"], 1, "CONDITIONING"),
        (ids["vae_loader"], 0, ids["i2v_latent"], 2, "VAE"),
        (ids["select_image"], 0, ids["image_scale"], 0, "IMAGE"),
        (ids["image_scale"], 0, ids["i2v_latent"], 4, "IMAGE"),
        (ids["i2v_latent"], 0, ids["ksampler_high"], 1, "CONDITIONING"),
        (ids["i2v_latent"], 0, ids["ksampler_low"], 1, "CONDITIONING"),
        (ids["i2v_latent"], 1, ids["ksampler_high"], 2, "CONDITIONING"),
        (ids["i2v_latent"], 1, ids["ksampler_low"], 2, "CONDITIONING"),
        (ids["i2v_latent"], 2, ids["ksampler_high"], 3, "LATENT"),
        (ids["ksampler_high"], 0, ids["vram_cleanup_mid"], 0, "*"),
        (ids["vram_cleanup_mid"], 0, ids["ram_cleanup_mid"], 0, "*"),
        (ids["ram_cleanup_mid"], 0, ids["ksampler_low"], 3, "LATENT"),
        (ids["ksampler_low"], 0, ids["vae_decode"], 0, "LATENT"),
        (ids["vae_loader"], 0, ids["vae_decode"], 1, "VAE"),
        (ids["vae_decode"], 0, ids["vram_cleanup_final"], 0, "*"),
        (ids["vram_cleanup_final"], 0, ids["ram_cleanup_final"], 0, "*"),
        (prev_turn_output_node_id, 0, ids["select_image"], 0, "IMAGE"),
        (prev_turn_output_node_id, 0, ids["batch_images"], 0, "IMAGE"),
        (ids["ram_cleanup_final"], 0, ids["batch_images"], 1, "IMAGE"),
        (ids["vae_decode"], 0, ids["turn_video"], 0, "IMAGE"),
    ]

    return nodes, link_defs, ids["batch_images"]


def generate_workflow(script_path, turns_range=None, image_path=None):
    print(f"Loading movie script: {script_path}")

    try:
        with open(script_path, "r", encoding="utf-8") as f:
            movie_script = json.load(f)
    except Exception as e:
        print(f"Error loading script file: {e}")
        raise

    workflow_name = os.path.splitext(os.path.basename(script_path))[0]

    script_turns = movie_script.get("turns", {})
    available_turns = [int(k) for k in script_turns.keys() if k.isdigit()]

    if not available_turns:
        raise ValueError("Movie script contains no valid turns.")

    if turns_range is None:
        selected_turns = sorted(available_turns)
    else:
        selected_turns = []
        for turn_num in turns_range:
            if turn_num not in available_turns:
                raise ValueError(
                    f"Turn {turn_num} not found in script. Available turns: {available_turns}"
                )
            selected_turns.append(turn_num)
        selected_turns = sorted(selected_turns)

    if image_path:
        image_filename = os.path.basename(image_path)
        if not os.path.exists(image_path):
            raise ValueError(f"Image file not found: {image_path}")
        print(f"Using provided image: {image_filename}")
        print("First turn will use I2V with image upscaling instead of T2V")

    print(f"Processing turns: {selected_turns}")
    print("Enhanced with multiple LoRAs: Lightning + Optional")
    print(f"Image upscaling: {UPSCALE_METHOD} method at {UPSCALE_FACTOR}x scale")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    turns_str = (
        f"{min(selected_turns)}-{max(selected_turns)}"
        if len(selected_turns) > 1
        else str(selected_turns[0])
    )

    all_nodes, all_link_defs = [], []
    last_turn_output_node_id = None
    is_first_turn = True

    for turn_num in selected_turns:
        turn_data = script_turns[str(turn_num)]
        print(f"Generating nodes for Turn {turn_num}...")

        positive_prompt = turn_data.get("positive_prompt", "")
        negative_prompt = turn_data.get("negative_prompt", "")

        if is_first_turn:
            if image_path:
                image_filename = os.path.basename(image_path)
                nodes, link_defs, output_node_id = create_first_turn_i2v(
                    turn_num,
                    positive_prompt,
                    negative_prompt,
                    image_filename,
                    workflow_name,
                )
            else:
                nodes, link_defs, output_node_id = create_t2v_turn(
                    turn_num, positive_prompt, negative_prompt, workflow_name
                )
            is_first_turn = False
        else:
            if last_turn_output_node_id is None:
                raise RuntimeError(
                    "Cannot create I2V turn; previous turn's output is missing."
                )
            nodes, link_defs, output_node_id = create_i2v_turn(
                turn_num,
                positive_prompt,
                negative_prompt,
                last_turn_output_node_id,
                workflow_name,
            )

        all_nodes.extend(nodes)
        all_link_defs.extend(link_defs)
        last_turn_output_node_id = output_node_id

    print("Adding final video combination node...")
    final_combine_id = max(selected_turns) * NODE_ID_BASE_OFFSET + 1
    final_combine_pos = [max(selected_turns) * HORIZONTAL_SPACING, 300]
    final_combine_node = create_node(
        "VHS_VideoCombine",
        final_combine_id,
        final_combine_pos,
        [
            FRAME_RATE,
            VIDEO_LOOP_COUNT,
            f"{workflow_name}_turns{turns_str}_base_{timestamp}",
            VIDEO_FORMAT,
            VIDEO_PINGPONG,
            VIDEO_SAVE_OUTPUT,
            VIDEO_PIXEL_FORMAT,
            VIDEO_CRF,
            VIDEO_SAVE_METADATA,
            VIDEO_TRIM_TO_AUDIO,
        ],
    )
    all_nodes.append(final_combine_node)
    all_link_defs.append((last_turn_output_node_id, 0, final_combine_id, 0, "IMAGE"))

    print("Applying all connections...")
    final_links = []
    nodes_by_id = {node["id"]: node for node in all_nodes}
    link_id_counter = 1

    for link_def in all_link_defs:
        if len(link_def) != 5:
            print(f"Warning: Invalid link definition: {link_def}")
            continue

        origin_id, origin_slot, target_id, target_slot, link_type = link_def

        final_links.append(
            [link_id_counter, origin_id, origin_slot, target_id, target_slot, link_type]
        )

        origin_node = nodes_by_id.get(origin_id)
        target_node = nodes_by_id.get(target_id)

        if not origin_node:
            print(f"Warning: Origin node {origin_id} not found for link")
            continue

        if not target_node:
            print(f"Warning: Target node {target_id} not found for link")
            continue

        if "outputs" not in origin_node or origin_slot >= len(origin_node["outputs"]):
            print(f"Warning: Invalid origin slot {origin_slot} for node {origin_id}")
            continue

        if origin_node["outputs"][origin_slot].get("links") is None:
            origin_node["outputs"][origin_slot]["links"] = []
        origin_node["outputs"][origin_slot]["links"].append(link_id_counter)

        if "inputs" not in target_node or target_slot >= len(target_node["inputs"]):
            print(f"Warning: Invalid target slot {target_slot} for node {target_id}")
            continue

        target_node["inputs"][target_slot]["link"] = link_id_counter

        link_id_counter += 1

    print(f"Generated {len(all_nodes)} nodes and {len(final_links)} links")
    print(f"Processing turns: {selected_turns}")
    print("Multi-LoRA configuration: Lightning + Optional applied to all turns")
    print(f"Image upscaling: {UPSCALE_METHOD} @ {UPSCALE_FACTOR}x for I2V inputs")
    print("Pipeline: T2V/I2V Generation → Final Combined Video")
    print(
        f"Final output: {workflow_name}_turns{turns_str}_base_{timestamp}.mp4 at {VIDEO_WIDTH}x{VIDEO_HEIGHT}@{FRAME_RATE}fps"
    )

    final_workflow = {
        "id": str(uuid.uuid4()),
        "revision": 0,
        "last_node_id": max(n["id"] for n in all_nodes) if all_nodes else 0,
        "last_link_id": max(link[0] for link in final_links) if final_links else 0,
        "nodes": all_nodes,
        "links": final_links,
        "groups": [],
        "config": {},
        "extra": {"ds": {"scale": 0.7, "offset": [0, 0]}, "frontendVersion": "1.24.4"},
        "version": 0.4,
    }

    return final_workflow


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a ComfyUI video workflow from a movie script."
    )
    parser.add_argument(
        "script_path", type=str, help="Path to the movie script JSON file."
    )
    parser.add_argument(
        "--turns",
        type=str,
        help="Specify turns to process (e.g., '3' for turn 3, '1:4' for turns 1-4)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to image file for first turn I2V (overrides T2V for first turn)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.script_path):
        print(f"Error: Script file not found at '{args.script_path}'", file=sys.stderr)
        sys.exit(1)

    turns_range = None
    if args.turns:
        try:
            turns_range = parse_turns_range(args.turns)
        except ValueError as e:
            print(f"Error parsing turns range '{args.turns}': {e}", file=sys.stderr)
            sys.exit(1)

    try:
        new_workflow = generate_workflow(args.script_path, turns_range, args.image)
        base_script_name = os.path.splitext(os.path.basename(args.script_path))[0]
        turns_suffix = f"_turns_{args.turns.replace(':', '-')}" if args.turns else ""
        image_suffix = "_i2v" if args.image else ""
        output_filename = (
            f"{base_script_name}{turns_suffix}{image_suffix}_workflow.json"
        )

        with open(output_filename, "w", encoding="utf-8") as f:
            json.dump(new_workflow, f, indent=2)

        print("\n" + "=" * 50)
        print(
            "✅ Success! Video generation workflow with image upscaling and multi-LoRA support generated!"
        )
        print(f"Saved as: {output_filename}")
        print(
            f"Generated {len(new_workflow['nodes'])} nodes and {len(new_workflow['links'])} links"
        )
        if args.turns:
            print(f"Processed turns: {args.turns}")
        if args.image:
            print(f"First turn initialized with image: {os.path.basename(args.image)}")
        print("\nImage upscaling configuration:")
        print(f"- Method: {UPSCALE_METHOD}")
        print(f"- Scale factor: {UPSCALE_FACTOR}x")
        print("- Applied to all I2V input images")
        print("\nApplied LoRAs to all turns:")

        enabled_high = [
            config
            for config in T2V_HIGH_NOISE_LORA.values()
            if config.get("enabled", True)
        ]
        enabled_low = [
            config
            for config in T2V_LOW_NOISE_LORA.values()
            if config.get("enabled", True)
        ]

        print(f"- High noise LoRAs: {len(enabled_high)} enabled")
        print(f"- Low noise LoRAs: {len(enabled_low)} enabled")
        print("\nOutput files:")
        print(f"- Per-turn videos: {base_script_name}_turn<N>.mp4")
        print(
            f"- Combined video: {base_script_name}_turns{args.turns or 'all'}_base_<timestamp>.mp4"
        )
        print("\nVideo specifications:")
        print(f"- Resolution: {VIDEO_WIDTH}x{VIDEO_HEIGHT}")
        print(f"- Frame rate: {FRAME_RATE}fps")
        print(f"- Length: {VIDEO_LENGTH} frames")
        print("=" * 50)

    except Exception as e:
        print(f"\nAn error occurred: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
