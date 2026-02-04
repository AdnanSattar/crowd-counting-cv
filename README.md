# FIDTM - Crowd Localization and Counting

[![Python](https://img.shields.io/badge/Python-3.6+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.4+-orange.svg)](https://pytorch.org/)

FIDTM (Focal Inverse Distance Transform Maps) for crowd localization and counting using HR-Net (High-Resolution Network). This implementation detects and counts people in images and videos using deep learning.

## Project Overview

This project implements FIDTM-based crowd counting and localization, which is particularly effective for dense crowd scenarios. The model uses HR-Net architecture to generate density maps and localize individuals in crowded scenes.

**Note**: This is a density-based counting model optimized for accurate crowd counting. While it can generate bounding boxes, they are derived from density map peaks and may not be perfectly accurate for precise localization. For best results, use the count output rather than bounding boxes.

## Key Features

- **Crowd counting**: Accurate estimation of people count in images/videos
- **Density map generation**: Visual representation of crowd density
- **Multiple dataset support**: ShanghaiTech A/B, JHU-Crowd++, UCF-QNRF, NWPU
- **Video demo**: Real-time processing on video streams with optimized performance
- **GPU/CPU support**: Automatic device detection with CUDA support

## Demo

### Sample Output

The model processes videos and images, providing:

- **Crowd count**: Accurate estimation displayed on the output
- **Density visualization**: Optional density map overlay (disabled by default in video demo)

![Demo Output](image/demo.jpeg)

*Example: Crowd counting on a dense crowd scene. The green text shows the estimated count.*

## Project Structure

```
.
├── train_baseline.py          # Training script
├── test.py                     # Testing/evaluation script
├── video_demo.py               # Real-time video processing demo
├── make_npydata.py             # Dataset preparation (creates .npy file lists)
├── dataset.py                   # PyTorch dataset loader
├── config.py                    # Configuration and command-line arguments
├── image.py                     # Image processing utilities
├── utils.py                     # Utility functions
├── Networks/
│   └── HR_Net/                  # HR-Net (High-Resolution Network) architecture
│       ├── seg_hrnet.py         # HR-Net model definition
│       └── config.py            # HR-Net configuration
├── data/                        # Scripts to generate FIDT maps for different datasets
│   ├── fidt_generate_sh.py      # ShanghaiTech FIDT map generation
│   ├── fidt_generate_jhu.py     # JHU-Crowd++ FIDT map generation
│   └── fidt_generate_qnrf.py    # UCF-QNRF FIDT map generation
├── local_eval/                  # Evaluation scripts for localization metrics
│   ├── eval.py                 # Evaluation script
│   ├── eval_qnrf.py            # QNRF-specific evaluation
│   └── gt_generate.py           # Ground truth generation
├── dataset/                     # Dataset directory (not included in repo)
├── save_file/                   # Model checkpoints (not included in repo)
├── image/                       # Output images (not included in repo)
├── video/                       # Input videos (not included in repo)
├── requirements.txt             # Python dependencies
├── CUDA_SETUP.md                # CUDA installation guide
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # This file
```

## What this codebase has

- FIDTM (Focal Inverse Distance Transform Maps) for crowd localization and counting
- HR-Net architecture implementation
- Training and testing scripts
- Video demo for real-time processing with performance optimizations
- Support for multiple datasets: ShanghaiTech A/B, JHU-Crowd++, UCF-QNRF, NWPU
- Evaluation tools for localization metrics
- GPU/CPU support with automatic device detection
- Optimized video processing for large files

## Citation

If you use this code in your research, please cite the original FIDTM paper:

```bibtex
@article{fidtm2021,
  title={FIDTM: Focal Inverse Distance Transform Maps for Crowd Localization and Counting},
  author={...},
  journal={...},
  year={2021}
}
```

## License

[Add your license here]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Original FIDTM implementation
- HR-Net architecture
- Dataset providers: ShanghaiTech, JHU-Crowd++, UCF-QNRF, NWPU

## Dependencies

- Python >= 3.6
- PyTorch >= 1.4 (with CUDA support recommended)
- opencv-python >= 4.0
- scipy >= 1.4.0
- h5py >= 2.10
- pillow >= 7.0.0
- imageio >= 1.18
- nni >= 2.0
- torchvision >= 0.5.0
- yacs
- easydict
- numpy

See `requirements.txt` for the complete list.

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd fidtm-crowd-counting
```

### 2. Create a virtual environment (recommended)

Using `uv` (recommended):

```bash
uv venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

Or using `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Or using UV package manager:

```bash
uv pip install -r requirements.txt
```

### 4. CUDA Setup (Optional, for GPU acceleration)

If you have an NVIDIA GPU, install CUDA-enabled PyTorch:

```bash
# For CUDA 12.4
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

See `CUDA_SETUP.md` for detailed CUDA installation instructions.

## Usage

### Dataset Preparation

1. Place your datasets in the `dataset/` directory with the following structure:
   - `dataset/ShanghaiTech/` - ShanghaiTech dataset
   - `dataset/jhu_crowd_v2.0/` - JHU-Crowd++ dataset
   - `dataset/UCF-QNRF_ECCV18/` - UCF-QNRF dataset
   - `dataset/NWPU_localization/` - NWPU-Crowd dataset (optional)

#### Dataset Download Links

- **ShanghaiTech**: Available from the original paper or various academic repositories
- **JHU-Crowd++**: Available from [JHU-CROWD++ Dataset](http://www.crowd-counting.com/)
- **UCF-QNRF**: Available from [UCF-QNRF Dataset](https://www.crcv.ucf.edu/data/ucf-qnrf/)
- **NWPU-Crowd** (Optional):
  - Official download: <https://crowdbenchmark.com/nwpucrowd.html> (registration required)
  - Sample code & resources: <https://gjy3035.github.io/NWPU-Crowd-Sample-Code/>
  - **Note**: NWPU dataset is optional. The project works fine without it. If you want to use it, you also need to create `data/NWPU_list/` directory with `train.txt`, `val.txt`, and `test.txt` files containing image filenames

1. Run `make_npydata.py` to generate .npy files:

```bash
python make_npydata.py
```

### Training

For training, run `train_baseline.py` with appropriate arguments:

```bash
python train_baseline.py --dataset ShanghaiA --gpu_id 0
```

### Testing

For testing, run `test.py` with a pre-trained model:

```bash
python test.py --dataset ShanghaiA --pre ./model_best.pth --gpu_id 0
```

### Video Demo

For video demo, run `video_demo.py` with a video path:

```bash
python video_demo.py --video_path ./video/video1_24fps.mp4 --pre ./save_file/A_baseline/model_best.pth --gpu_id 0
```

**Note**: The video demo is optimized for performance:

- Automatically downscales large videos (4K → 960p) for faster processing
- Processes every 2nd frame for long videos to reduce computation time
- Shows crowd count overlay on the original video
- Bounding boxes are disabled by default (set `SHOW_BOXES = True` in `video_demo.py` to enable, but note they may be inaccurate)

**Example Output**:

- Input: Video file (e.g., `./video/video1_24fps.mp4`)
- Output: `./demo.avi` with crowd count overlay
- Snapshot: `./image/demo.jpeg` (saved every 10 frames)

## Model Limitations

- **Counting Accuracy**: The model is optimized for counting accuracy and performs well on dense crowds
- **Localization Accuracy**: Bounding boxes are derived from density map peaks and may not perfectly align with individual people. For precise localization, consider using detection-based models (YOLO, Faster R-CNN, etc.)
- **Best Use Case**: Dense crowd counting where precise individual localization is not required
