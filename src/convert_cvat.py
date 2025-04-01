import os
import cv2
import numpy as np
from tqdm import tqdm

from argparse import ArgumentParser


def convert_cvat_masks_to_binary(input_dir, output_dir, threshold=1):
    """
    Converts masks exported from CVAT to binary segmentation masks.

    Args:
        input_dir (str): Path to the directory containing CVAT-exported masks.
        output_dir (str): Path to save the binary masks.
        threshold (int): Threshold to binarize the mask (default: 1, treating all non-zero pixels as foreground).
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_files = [f for f in os.listdir(input_dir) if f.endswith('.png')]

    for mask_file in tqdm(mask_files, desc="Converting masks"):
        mask_path = os.path.join(input_dir, mask_file)
        binary_mask_path = os.path.join(output_dir, mask_file)

        # Load the mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Convert to binary mask
        binary_mask = (mask >= threshold).astype(np.uint8) * 255

        # Save the binary mask
        cv2.imwrite(binary_mask_path, binary_mask)

    print(f"Converted {len(mask_files)} masks and saved to {output_dir}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--input_folder", help="Input folder containing masks from cvat")
    parser.add_argument("--output_folder", help="Output folder to save masks")
    args = parser.parse_args()
    convert_cvat_masks_to_binary(args.input_folder, args.output_folder)
