# import pandas as pd
# import os

# input_folder = "//media/divy/5c85a034-6379-4cd7-a75b-c4ed0b105d26/Jay_Casadi/run24/20260528_170812"        # folder containing your CSV files
# output_folder = "//media/divy/5c85a034-6379-4cd7-a75b-c4ed0b105d26/Jay_Casadi/run24/20260528_170812"  # folder where .txt files will be saved

# os.makedirs(output_folder, exist_ok=True)

# for filename in os.listdir(input_folder):
#     if filename.endswith(".csv"):
#         csv_path = os.path.join(input_folder, filename)
#         txt_filename = filename.replace(".csv", ".txt")
#         txt_path = os.path.join(output_folder, txt_filename)

#         df = pd.read_csv(csv_path)
#         with open(txt_path, 'w') as f:
#             f.write(df.to_string(index=False))

#         print(f"Converted: {filename} → {txt_filename}")

# print("All done!")
"""
Batch Depth Estimator
---------------------
Converts all RGB images in a 'frames/' subfolder to depth maps,
saving them into a 'depth/' subfolder at the same level.

Usage:
    python batch_depth_estimator.py /path/to/run_folder
    python batch_depth_estimator.py /media/divy/5c85a034-6379-4cd7-a75b-c4ed0b105d26/Jay_Casadi/run18

The script expects:
    run_folder/
        frames/   <-- RGB images (jpg, png, etc.)
    
It will create:
    run_folder/
        frames/
        depth/    <-- Depth maps saved as PNG (same base filenames)
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from transformers import pipeline


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_pipeline(device):
    print(f"Loading Depth-Anything-V2-Small model on {'GPU' if device == 0 else 'CPU'}...")
    pipe = pipeline(
        task="depth-estimation",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device=device
    )
    print("Model loaded.\n")
    return pipe


def get_depth_map(pipe, image_path: Path) -> np.ndarray:
    """
    Run depth estimation on a single image.
    Returns depth map normalized to 0-255 uint8.
    """
    image = Image.open(image_path).convert("RGB")
    output = pipe(image)
    depth = output["depth"]

    if not isinstance(depth, np.ndarray):
        depth = np.array(depth)

    depth = depth.astype(np.float32)
    depth_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255.0
    return depth_norm.astype(np.uint8)


def process_folder(run_folder: str):
    run_path = Path(run_folder)
    frames_path = run_path / "frames"
    depth_path = run_path / "depth"

    # Validate input folder
    if not run_path.exists():
        print(f"[ERROR] Run folder does not exist: {run_path}")
        sys.exit(1)
    if not frames_path.exists():
        print(f"[ERROR] 'frames' subfolder not found in: {run_path}")
        sys.exit(1)

    # Collect all image files
    image_files = sorted([
        f for f in frames_path.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(f"[ERROR] No supported images found in: {frames_path}")
        print(f"        Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    print(f"Found {len(image_files)} images in: {frames_path}")

    # Create depth output folder
    depth_path.mkdir(parents=True, exist_ok=True)
    print(f"Depth maps will be saved to: {depth_path}\n")

    # Load model
    device = 0 if torch.cuda.is_available() else -1
    pipe = load_pipeline(device)

    # Process images
    success_count = 0
    fail_count = 0

    for idx, img_path in enumerate(image_files, start=1):
        out_filename = img_path.stem + "_depth.png"
        out_path = depth_path / out_filename

        # Skip already-processed files
        if out_path.exists():
            print(f"[{idx}/{len(image_files)}] Skipping (already exists): {out_filename}")
            success_count += 1
            continue

        try:
            depth_map = get_depth_map(pipe, img_path)
            Image.fromarray(depth_map).save(out_path)
            print(f"[{idx}/{len(image_files)}] Saved: {out_filename}")
            success_count += 1
        except Exception as e:
            print(f"[{idx}/{len(image_files)}] FAILED: {img_path.name} -> {e}")
            fail_count += 1

    print(f"\nDone! {success_count} succeeded, {fail_count} failed.")
    print(f"Depth maps saved in: {depth_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch convert RGB images to depth maps using Depth-Anything-V2."
    )
    parser.add_argument(
        "run_folder",
        type=str,
        help="Path to the run folder containing a 'frames/' subfolder."
    )
    args = parser.parse_args()
    process_folder(args.run_folder)


if __name__ == "__main__":
    main()