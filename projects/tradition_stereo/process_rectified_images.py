"""
Process rectified images with three operations:
1. Interpolate to 2x resolution
2. Swap left/right and rotate 180 degrees
3. Swap left/right, rotate 180 degrees, and interpolate to 2x resolution
"""

import cv2
import os
from pathlib import Path
from tqdm import tqdm


def process_images(input_root, output_roots):
    """
    Process all image pairs in input_root and save to three output folders.

    Args:
        input_root: Path to rectified_images folder
        output_roots: Dict with keys '2x', 'swapped_flipped', 'swapped_flipped_2x'
    """
    input_path = Path(input_root)

    # Get all subdirectories containing image pairs
    scene_folders = [f for f in input_path.iterdir() if f.is_dir()]

    print(f"Found {len(scene_folders)} scene folders to process")

    for scene_folder in tqdm(scene_folders, desc="Processing scenes"):
        scene_name = scene_folder.name

        # Check if both im0.png and im1.png exist
        im0_path = scene_folder / "im0.png"
        im1_path = scene_folder / "im1.png"

        if not im0_path.exists() or not im1_path.exists():
            print(f"Warning: Skipping {scene_name}, missing im0.png or im1.png")
            continue

        # Read images
        im0 = cv2.imread(str(im0_path))
        im1 = cv2.imread(str(im1_path))

        if im0 is None or im1 is None:
            print(f"Warning: Failed to read images in {scene_name}")
            continue

        h, w = im0.shape[:2]

        # Create output directories for this scene
        for key, output_root in output_roots.items():
            output_scene = Path(output_root) / scene_name
            output_scene.mkdir(parents=True, exist_ok=True)

        # Operation 1: Interpolate to 2x resolution
        im0_2x = cv2.resize(im0, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        im1_2x = cv2.resize(im1, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        output_scene_2x = Path(output_roots['2x']) / scene_name
        cv2.imwrite(str(output_scene_2x / "im0.png"), im0_2x)
        cv2.imwrite(str(output_scene_2x / "im1.png"), im1_2x)

        # Operation 2: Swap left/right and rotate 180 degrees
        # Swap: im0 becomes im1, im1 becomes im0
        # Rotate: rotate 180 degrees
        im0_swapped_rotated = cv2.rotate(im1, cv2.ROTATE_180)  # im1 → im0 (swapped and rotated)
        im1_swapped_rotated = cv2.rotate(im0, cv2.ROTATE_180)  # im0 → im1 (swapped and rotated)

        output_scene_sf = Path(output_roots['swapped_flipped']) / scene_name
        cv2.imwrite(str(output_scene_sf / "im0.png"), im0_swapped_rotated)
        cv2.imwrite(str(output_scene_sf / "im1.png"), im1_swapped_rotated)

        # Operation 3: Swap, rotate 180 degrees, and interpolate to 2x
        im0_sf_2x = cv2.resize(im0_swapped_rotated, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)
        im1_sf_2x = cv2.resize(im1_swapped_rotated, (w * 2, h * 2), interpolation=cv2.INTER_LINEAR)

        output_scene_sf2x = Path(output_roots['swapped_flipped_2x']) / scene_name
        cv2.imwrite(str(output_scene_sf2x / "im0.png"), im0_sf_2x)
        cv2.imwrite(str(output_scene_sf2x / "im1.png"), im1_sf_2x)

    print(f"\nProcessing complete!")
    print(f"Output folders:")
    for key, path in output_roots.items():
        print(f"  - {key}: {path}")


def main():
    # Input folder
    input_root = r"D:\Desktop\stereo_project\tradition_stereo\rectified_images"

    # Output folders
    base_dir = Path(input_root).parent
    output_roots = {
        '2x': str(base_dir / "rectified_images_2x"),
        'swapped_flipped': str(base_dir / "rectified_images_swapped_flipped"),
        'swapped_flipped_2x': str(base_dir / "rectified_images_swapped_flipped_2x")
    }

    # Process images
    process_images(input_root, output_roots)


if __name__ == "__main__":
    main()
