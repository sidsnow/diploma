import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    DDIMScheduler,
    AutoencoderKL,
    ControlNetModel,
)
from PIL import Image
import os
from tqdm import tqdm
from ip_adapter import IPAdapter
import numpy as np
import cv2

base_model_path = "runwayml/stable-diffusion-v1-5"
vae_model_path = "stabilityai/sd-vae-ft-mse"
image_encoder_path = "models/image_encoder/"
ip_ckpt = "ip_adapter.bin"
device = "cuda"

noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00095,
    beta_end=0.016,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)
vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=torch.float16)
# load SD pipeline
# load controlnet
controlnet_model_path = "lllyasviel/control_v11p_sd15_canny"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)
# load SD pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    scheduler=noise_scheduler,
    vae=vae,
    feature_extractor=None,
    safety_checker=None
)
# load ip-adapter
ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)

masks_path = r"..\data\generated_masks_2"
masks_files = os.listdir(masks_path)
images_path = r"..\data\data\covdor_generated"
image_files = os.listdir(images_path)


save_path = r"..\data\generated"

tqdm_iterator = tqdm(zip(image_files, masks_files))
for imfile, maskfile in tqdm_iterator:

    tqdm_iterator.set_description(imfile)
    image = Image.open(os.path.join(images_path, imfile))
    height = 512
    width = int(800 / 640 * height)
    image = image.resize((height, width))

    g_image = Image.open(os.path.join(masks_path, maskfile))
    g_image = g_image.convert('RGB')
    g_image = g_image.resize(image.size)

    g_image = np.array(g_image)
    canny_image = cv2.Canny(g_image, 60, 100)

    prompt_image = Image.fromarray(canny_image)
    # generate image variations
    images = ip_model.generate(
        pil_image=image,
        image=prompt_image,
        num_samples=1,
        num_inference_steps=50,
        seed=42,
        negative_prompt="best quality, regular shapes, even surface, round, metallic, slippery, polished, smooth",
        prompt="conveyer belt, monochrome, worst quality, poor lightning, rough, porous, angular, rocks",
        width=image.size[0],
        height=image.size[1],
        scale=1.5,
        guidance_scale=1,
        guess_mode=True,
        controlnet_conditioning_scale=4.0,
        )
    
    for idx, im in enumerate(images):
        im.save(
            os.path.join(save_path, maskfile)
        )