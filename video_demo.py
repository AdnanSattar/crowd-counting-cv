from __future__ import division

import logging
import math
import warnings

import nni
import torch.nn as nn
import torch.nn.functional as F
from nni.utils import merge_parameter
from torchvision import datasets, transforms

import dataset
from config import args, return_args
from image import *
from Networks.HR_Net.seg_hrnet import get_seg_model
from utils import *

warnings.filterwarnings("ignore")
import time

logger = logging.getLogger("mnist_AutoML")

print(args)
img_transform = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)
tensor_transform = transforms.ToTensor()


def main(args):
    model = get_seg_model()
    use_cuda = torch.cuda.is_available() and args["gpu_id"] != "cpu"
    if use_cuda:
        model = nn.DataParallel(model, device_ids=[0])
        model = model.cuda()
        device = "cuda"
    else:
        print("Using CPU for video demo")
        device = "cpu"

    if args["pre"]:
        if os.path.isfile(args["pre"]):
            print("=> loading checkpoint '{}'".format(args["pre"]))
            checkpoint = torch.load(args["pre"])
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            args["start_epoch"] = checkpoint["epoch"]
            args["best_pred"] = checkpoint["best_prec1"]
        else:
            print("=> no checkpoint found at '{}'".format(args["pre"]))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

    cap = cv2.VideoCapture(args["video_path"])
    ret, frame = cap.read()
    print(f"Input video shape: {frame.shape}")

    # Resize for faster processing - use smaller scale for 4K videos
    # More aggressive downscaling for speed
    if frame.shape[1] > 2000:
        scale_factor = 0.25  # Downscale 4K to ~960x540
    elif frame.shape[1] > 1500:
        scale_factor = 0.5
    else:
        scale_factor = 1.0
    if scale_factor < 1.0:
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        print(f"Resized to: {frame.shape} for faster processing")

    """out video - use smaller output resolution for faster writing"""
    # Output at processed resolution (not original 4K) for speed
    width = frame.shape[1]
    height = frame.shape[0]
    # Use MP4V codec for better compression and speed
    # Output at original size (no need for larger output since we're not stacking images)
    out_width = width
    out_height = height
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("./demo.avi", fourcc, 24.0, (out_width, out_height))

    # Get total frame count for progress tracking
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    import time

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print(
                f"\nVideo processing complete! Processed {frame_idx}/{total_frames} frames"
            )
            break

        frame_idx += 1
        # Show progress every 10 frames
        if frame_idx % 10 == 0:
            elapsed = time.time() - start_time
            fps_processed = frame_idx / elapsed if elapsed > 0 else 0
            remaining_frames = total_frames - frame_idx
            eta_seconds = remaining_frames / fps_processed if fps_processed > 0 else 0
            print(
                f"\nProgress: {frame_idx}/{total_frames} frames ({100*frame_idx/total_frames:.1f}%) | "
                f"Speed: {fps_processed:.2f} fps | ETA: {eta_seconds/60:.1f} minutes"
            )

        # Frame already resized above, just resize if needed (for consistency)
        if scale_factor < 1.0:
            frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)
        ori_img = frame.copy()
        frame = frame.copy()
        image = tensor_transform(frame)
        image = img_transform(image).unsqueeze(0)
        if use_cuda:
            image = image.cuda()
        else:
            image = image

        with torch.no_grad():
            d6 = model(image)

            # Use higher threshold (0.5 = 50% of max) to reduce false positives
            # This improves bounding box accuracy by filtering out low-confidence detections
            count, pred_kpoint = counting(d6, threshold_ratio=0.50)

            # Option: Set SHOW_BOXES to False to disable bounding boxes and only show count
            # Bounding boxes from density-based models are often inaccurate for localization
            SHOW_BOXES = False  # Set to False to disable boxes

            # Generate bounding boxes on original image
            # For very large crowds, skip some frames to speed up
            generate_boxes = SHOW_BOXES and (
                (frame_idx % 5 == 0 or frame_idx <= 5 or frame_idx > total_frames - 5)
                or count < 5000  # Always generate boxes for smaller crowds
            )

            # Start with original image
            res = ori_img.copy()

            # Add red bounding boxes if enabled
            if generate_boxes:
                res = generate_bounding_boxes(pred_kpoint, res)

            # Add count text (in green for visibility)
            cv2.putText(
                res,
                "Count:" + str(int(count)),
                (30, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),  # Green text for visibility
                2,
            )

            # Ensure output matches video dimensions
            if res.shape[1] != out_width or res.shape[0] != out_height:
                res = cv2.resize(res, (out_width, out_height))
            os.makedirs("./image", exist_ok=True)
            # Only save image every 10 frames to reduce I/O overhead
            if frame_idx % 10 == 0:
                cv2.imwrite("./image/demo.jpeg", res)
            """write in out_video"""
            # Resize to output dimensions if needed
            if res.shape[1] != out_width or res.shape[0] != out_height:
                res = cv2.resize(res, (out_width, out_height))
            out.write(res)
            last_res = res  # Store for frame skipping

        # Print count less frequently to reduce console spam
        if frame_idx % 5 == 0:
            print(f"Frame {frame_idx}: pred:{count:.3f}", end=" | ", flush=True)

    # Cleanup
    cap.release()
    out.release()
    print(f"\n\nVideo processing complete! Output saved to ./demo.avi")
    print(f"Processed {frame_idx}/{total_frames} frames")


def counting(input, threshold_ratio=0.5):
    """
    Extract count and keypoints from model output.

    Args:
        input: Model output tensor
        threshold_ratio: Threshold ratio (0.0-1.0). Higher = more strict, fewer false positives.
                         Default 0.5 (50% of max) for better accuracy.
    """
    input_max = torch.max(input).item()
    keep = nn.functional.max_pool2d(input, (3, 3), stride=1, padding=1)
    keep = (keep == input).float()
    input = keep * input

    # Use higher threshold to reduce false positives (was 100/255 ≈ 0.39, now configurable)
    threshold = threshold_ratio * input_max
    input[input < threshold] = 0
    input[input > 0] = 1

    """negative sample"""
    if input_max < 0.1:
        input = input * 0

    count = int(torch.sum(input).item())

    kpoint = input.data.squeeze(0).squeeze(0).cpu().numpy()

    return count, kpoint


def generate_point_map(kpoint):
    """Optimized point map generation"""
    rate = 1
    pred_coor = np.nonzero(kpoint)
    point_map = (
        np.zeros(
            (int(kpoint.shape[0] * rate), int(kpoint.shape[1] * rate), 3), dtype="uint8"
        )
        + 255
    )

    # Optimize: Limit number of points drawn for very large crowds
    max_points = 1000
    num_points = len(pred_coor[0])
    if num_points > max_points:
        # Sample points uniformly
        indices = np.linspace(0, num_points - 1, max_points, dtype=int)
        h_coords = pred_coor[0][indices]
        w_coords = pred_coor[1][indices]
    else:
        h_coords = pred_coor[0]
        w_coords = pred_coor[1]

    # Draw points more efficiently
    for i in range(len(h_coords)):
        h = int(h_coords[i] * rate)
        w = int(w_coords[i] * rate)
        cv2.circle(point_map, (w, h), 2, (0, 0, 0), -1)  # Smaller circles for speed

    return point_map


def generate_bounding_boxes(kpoint, Img_data):
    """
    Generate bounding boxes from keypoints.

    Note: This is a density-based counting model, not a detection model.
    Bounding boxes are derived from density map peaks and may not be perfectly accurate.
    For better localization accuracy, consider using a detection-based model.
    """
    pts = np.array(list(zip(np.nonzero(kpoint)[1], np.nonzero(kpoint)[0])))

    if pts.shape[0] == 0:
        return Img_data

    # Filter out points too close to image edges (often false positives)
    margin = 10
    valid_mask = (
        (pts[:, 0] > margin)
        & (pts[:, 0] < Img_data.shape[1] - margin)
        & (pts[:, 1] > margin)
        & (pts[:, 1] < Img_data.shape[0] - margin)
    )
    pts = pts[valid_mask]

    if pts.shape[0] == 0:
        return Img_data

    # Optimize: For very large crowds, skip bounding boxes entirely or use fixed size
    if pts.shape[0] > 1000:
        # For very large crowds, just draw a few sample boxes to show detection
        sample_indices = np.linspace(
            0, pts.shape[0] - 1, min(50, pts.shape[0]), dtype=int
        )
        pts = pts[sample_indices]
        fixed_sigma = min(Img_data.shape[0], Img_data.shape[1]) * 0.015  # Smaller boxes
        for pt in pts:
            Img_data = cv2.rectangle(
                Img_data,
                (int(pt[0] - fixed_sigma), int(pt[1] - fixed_sigma)),
                (int(pt[0] + fixed_sigma), int(pt[1] + fixed_sigma)),
                (0, 0, 255),  # Red color (BGR format)
                2,  # Thicker lines for visibility
            )
        return Img_data

    # For smaller crowds, use KDTree (but still limit)
    max_points = 300  # Further reduced for speed
    if pts.shape[0] > max_points:
        indices = np.linspace(0, pts.shape[0] - 1, max_points, dtype=int)
        pts = pts[indices]

    leafsize = min(512, pts.shape[0])  # Smaller leafsize for speed

    if pts.shape[0] > 0:
        # build kdtree
        tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)

        distances, locations = tree.query(pts, k=min(4, pts.shape[0]))
        for index, pt in enumerate(pts):
            if np.sum(kpoint) > 1 and len(distances[index]) > 3:
                sigma = (
                    distances[index][1] + distances[index][2] + distances[index][3]
                ) * 0.1
            else:
                sigma = np.average(np.array(kpoint.shape)) / 2.0 / 2.0
            sigma = min(
                sigma, min(Img_data.shape[0], Img_data.shape[1]) * 0.03
            )  # Smaller max size
            sigma = max(sigma, 3)

            Img_data = cv2.rectangle(
                Img_data,
                (int(pt[0] - sigma), int(pt[1] - sigma)),
                (int(pt[0] + sigma), int(pt[1] + sigma)),
                (0, 0, 255),  # Red color (BGR format)
                2,  # Thicker lines for visibility
            )

    return Img_data


def show_fidt_func(input):
    input[input < 0] = 0
    input = input[0][0]
    fidt_map1 = input
    fidt_map1 = fidt_map1 / np.max(fidt_map1) * 255
    fidt_map1 = fidt_map1.astype(np.uint8)
    fidt_map1 = cv2.applyColorMap(fidt_map1, 2)
    return fidt_map1


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    tuner_params = nni.get_next_parameter()
    logger.debug(tuner_params)
    params = vars(merge_parameter(return_args, tuner_params))
    print(params)

    main(params)
