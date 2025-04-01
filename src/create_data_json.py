import os
import json
import argparse


def create_json_from_folder(folder_path, output_file):
    data = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                full_path = os.path.join(root, file)
                data.append({"image_file": full_path, "text": ""})

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a JSON file from a folder of images.")
    parser.add_argument("--folder_path", type=str,
                        help="Path to the folder containing images")
    parser.add_argument("--output_file", type=str,
                        help="Path to the output JSON file")

    args = parser.parse_args()
    create_json_from_folder(args.folder_path, args.output_file)
    print(f"JSON file created: {args.output_file}")
