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
from src.ip_adapter import IPAdapter
import numpy as np
import cv2
from omegaconf import DictConfig


def create_dataset(cfg: DictConfig):
    model_cfg = cfg.model
    data_cfg = cfg.data

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00095,
        beta_end=0.016,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(
        model_cfg.vae_model_path).to(dtype=torch.float16)

    controlnet_model_path = model_cfg.controlnet_path
    controlnet = ControlNetModel.from_pretrained(
        controlnet_model_path, torch_dtype=torch.float16)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        model_cfg.base_model_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None
    )

    ip_model = IPAdapter(pipe, model_cfg.image_encoder_path,
                         model_cfg.ip_ckpt, model_cfg.device)

    masks_path = data_cfg.masks_path
    masks_files = os.listdir(masks_path)
    images_path = data_cfg.images_path
    image_files = os.listdir(images_path)

    save_path = data_cfg.save_path

    tqdm_iterator = tqdm(zip(image_files, masks_files))
    for imfile, maskfile in tqdm_iterator:

        tqdm_iterator.set_description(imfile)
        image = Image.open(os.path.join(images_path, imfile))
        height = data_cfg.height
        width = data_cfg.width
        image = image.resize((height, width))

        g_image = Image.open(os.path.join(masks_path, maskfile))
        g_image = g_image.convert('RGB')
        g_image = g_image.resize(image.size)

        g_image = np.array(g_image)
        canny_image = cv2.Canny(g_image, 60, 100)

        prompt_image = Image.fromarray(canny_image)

        images = ip_model.generate(
            pil_image=image,
            image=prompt_image,
            num_samples=1,
            num_inference_steps=50,
            seed=42,
            prompt=model_cfg.prompt.positive,
            negative_prompt=model_cfg.prompt.negative,
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


if __name__ == "__main__":
    create_dataset()
