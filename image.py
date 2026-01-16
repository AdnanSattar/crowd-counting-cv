import cv2
import h5py
import numpy as np
import scipy
import scipy.io as io
import scipy.spatial
from PIL import Image


def load_data_fidt(img_path, args, train=True):
    gt_path = img_path.replace(".jpg", ".h5").replace("images", "gt_fidt_map")
    img = Image.open(img_path).convert("RGB")

    max_retries = 3
    retry_count = 0
    while retry_count < max_retries:
        try:
            gt_file = h5py.File(gt_path, "r")
            k = np.asarray(gt_file["kpoint"])
            fidt_map = np.asarray(gt_file["fidt_map"])
            gt_file.close()
            break
        except (OSError, FileNotFoundError) as e:
            retry_count += 1
            if retry_count >= max_retries:
                print(
                    f"ERROR: Cannot load FIDT map after {max_retries} attempts: {gt_path}"
                )
                print(f"Image path: {img_path}")
                print(
                    "Please generate FIDT maps first by running: python data/fidt_generate_sh.py"
                )
                raise FileNotFoundError(
                    f"FIDT map not found: {gt_path}. Please generate FIDT maps first."
                )
            print(
                f"Warning: path is wrong, can not load {gt_path}, retry {retry_count}/{max_retries}"
            )
            import time

            time.sleep(0.5)  # Wait a bit before retry

    img = img.copy()
    fidt_map = fidt_map.copy()
    k = k.copy()

    return img, fidt_map, k
