import torch
from diffusers import StableDiffusionPipeline
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

prompt = (
    "A high-contrast binary segmentation mask of rocks."
    "The image is strictly black and white: white regions are rocks, black is the background. Small rocks are overlapped by medium and big ones."
    "Small, medium, irregular, random, various shapes, rounder, scattered, clear thin boundaries between the rocks and the background."
)

# Load the Stable Diffusion model (ensure you have a suitable checkpoint)
model_id = "stabilityai/sd-turbo"


# Load pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline.to("cuda")  # Use GPU for faster inference

def generate_images(num, folder="../data/generated_masks_2"):
    for i in range(num):
        if os.path.exists(os.path.join(folder, str(i) + ".png")):
            continue
        image = pipeline(
            prompt,
            height=1024,
            width=720,
            num_inference_steps=50,
            guidance_scale=15,
            negative_prompt="triangles, not close",
            seed=43
            ).images[0]

        img = np.array(image)
        res = cv2.dilate(img, kernel=cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)), iterations=1)
        _, res2 = cv2.threshold(res, 60, 255, cv2.THRESH_BINARY)
    
        cv2.imwrite(os.path.join(folder, str(i) + ".png"), res2)

generate_images(1500)