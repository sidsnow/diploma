import os
import cv2
import argparse
from pathlib import Path


def read_bboxes(txt_path):
    """Reads the bounding boxes with class labels and relative coordinates from the .txt file"""
    bboxes = []
    with open(txt_path, 'r') as f:
        for line in f:
            # Assuming each line has the format: class_id x_center y_center width height
            parts = list(map(float, line.strip().split()))
            class_id = int(parts[0])
            bbox = parts[1:]
            bboxes.append((class_id, bbox))
    return bboxes


def rescale_bboxes(bboxes, crop_x, crop_y, crop_width, crop_height, img_width, img_height):
    """Rescales the bounding boxes after cropping"""
    # Calculate the scale factors for width and height
    x_scale = crop_width / img_width
    y_scale = crop_height / img_height

    rescaled_bboxes = []
    for class_id, bbox in bboxes:
        # Relative coordinates: x_center, y_center, width, height
        x_center, y_center, width, height = bbox

        # Convert relative coordinates to absolute pixel values
        x_center_abs = x_center * img_width
        y_center_abs = y_center * img_height
        width_abs = width * img_width
        height_abs = height * img_height

        # Check if the bounding box is within the crop area
        if x_center_abs - width_abs / 2 >= crop_x + crop_width or x_center_abs + width_abs / 2 <= crop_x or \
           y_center_abs - height_abs / 2 >= crop_y + crop_height or y_center_abs + height_abs / 2 <= crop_y:
            continue  # Skip this bbox if it's outside the crop

        # Clip the bbox to the crop area (if part of it is outside)
        x_center_abs = max(x_center_abs, crop_x)
        y_center_abs = max(y_center_abs, crop_y)
        x_center_abs = min(x_center_abs, crop_x + crop_width)
        y_center_abs = min(y_center_abs, crop_y + crop_height)

        # Convert back to relative coordinates in the cropped region
        x_center_crop = (x_center_abs - crop_x) / crop_width
        y_center_crop = (y_center_abs - crop_y) / crop_height
        width_crop = width_abs / crop_width
        height_crop = height_abs / crop_height

        # Add the rescaled bounding box
        rescaled_bboxes.append((
            class_id,
            x_center_crop,
            y_center_crop,
            width_crop,
            height_crop
        ))

    return rescaled_bboxes


def crop_and_rescale(image_path, txt_path, crop_x, crop_y, crop_width, crop_height, output_image_path, output_txt_path):
    """Crops the image and rescales the bounding boxes"""
    # Read image
    image = cv2.imread(image_path)
    img_height, img_width = image.shape[:2]

    # Crop the image
    cropped_image = image[crop_y:crop_y+crop_height, crop_x:crop_x+crop_width]

    # Read bounding boxes from the text file
    bboxes = read_bboxes(txt_path)

    # Rescale the bounding boxes
    rescaled_bboxes = rescale_bboxes(
        bboxes, crop_x, crop_y, crop_width, crop_height, img_width, img_height)

    # Save the cropped image
    cv2.imwrite(output_image_path, cropped_image)

    # Save the rescaled bounding boxes with class labels
    with open(output_txt_path, 'w') as f:
        for class_id, x_min, y_min, x_max, y_max in rescaled_bboxes:
            f.write(f"{class_id} {x_min} {y_min} {x_max} {y_max}\n")


def process_folder(input_folder, output_folder, crop_x, crop_y, crop_width, crop_height):
    """Processes all image and txt pairs in the input folder"""
    # Create output folder if it doesn't exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    for image_name in os.listdir(input_folder):
        if image_name.endswith('.png'):
            # Define paths
            image_path = os.path.join(input_folder, image_name)
            txt_path = os.path.join(
                input_folder, image_name.replace('.png', '.txt'))

            # Check if corresponding .txt file exists
            if not os.path.exists(txt_path):
                print(f"Warning: {txt_path} not found. Skipping.")
                continue

            # Define output paths
            output_image_path = os.path.join(output_folder, image_name)
            output_txt_path = os.path.join(
                output_folder, image_name.replace('.png', '.txt'))

            # Crop and rescale
            crop_and_rescale(image_path, txt_path, crop_x, crop_y,
                             crop_width, crop_height, output_image_path, output_txt_path)
            print(f"Processed: {image_name}")


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Crop images and rescale bounding boxes")
    parser.add_argument(
        "--input_folder", help="Input folder containing the image and txt pairs")
    parser.add_argument(
        "--output_folder", help="Output folder to save the cropped images and updated bounding boxes")
    parser.add_argument("--crop_x", type=int, default=500,
                        help="X coordinate of the crop start")
    parser.add_argument("--crop_y", type=int, default=200,
                        help="Y coordinate of the crop start")
    parser.add_argument("--crop_width", type=int, default=700,
                        help="Width of the crop region")
    parser.add_argument("--crop_height", type=int,
                        default=1000, help="Height of the crop region")

    args = parser.parse_args()

    process_folder(args.input_folder, args.output_folder,
                   args.crop_x, args.crop_y, args.crop_width, args.crop_height)
