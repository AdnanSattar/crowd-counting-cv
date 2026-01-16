# CUDA Setup Guide for PyTorch GPU Support

## Prerequisites

1. **NVIDIA GPU** - Check if you have one:

   ```powershell
   nvidia-smi
   ```

   If this command works, you have an NVIDIA GPU and drivers installed.

2. **Check CUDA Version**:

   ```powershell
   nvidia-smi
   ```

   Look for "CUDA Version: X.X" in the output. This tells you the maximum CUDA version your driver supports.

## Installation Steps

### Step 1: Install/Update NVIDIA Drivers

1. Visit: <https://www.nvidia.com/Download/index.aspx>
2. Enter your GPU model and download the latest driver
3. Install the driver and restart your computer

### Step 2: Check CUDA Compatibility

After installing drivers, check supported CUDA version:

```powershell
nvidia-smi
```

### Step 3: Install CUDA-Enabled PyTorch

**Option A: Using UV (Recommended - what you're using)**

1. Activate your virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Uninstall CPU-only PyTorch:

   ```powershell
   uv pip uninstall torch torchvision -y
   ```

3. Install CUDA-enabled PyTorch based on your CUDA version:

   **For CUDA 12.1:**

   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

   **For CUDA 11.8:**

   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
   ```

   **For CUDA 12.4:**

   ```powershell
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
   ```

**Option B: Using pip**

1. Activate virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

2. Uninstall CPU-only:

   ```powershell
   pip uninstall torch torchvision -y
   ```

3. Install CUDA version (example for CUDA 12.1):

   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

### Step 4: Verify CUDA Installation

Test if CUDA is working:

```powershell
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('GPU count:', torch.cuda.device_count()); print('GPU name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected output if successful:

```
CUDA available: True
CUDA version: 12.1
GPU count: 1
GPU name: NVIDIA GeForce RTX ...
```

### Step 5: Run Training with GPU

Once CUDA is working, you can use GPU:

```powershell
python train_baseline.py --dataset ShanghaiA --gpu_id 0 --epochs 1 --batch_size 4 --workers 4
```

## Common Issues

### Issue 1: "CUDA out of memory"

- **Solution**: Reduce batch size: `--batch_size 1` or `--batch_size 2`

### Issue 2: "CUDA version mismatch"

- **Solution**: Make sure PyTorch CUDA version matches your driver's supported version

### Issue 3: "No CUDA GPUs are available"

- **Solution**:
  - Check `nvidia-smi` works
  - Verify GPU is not being used by another process
  - Restart computer after driver installation

## Notes

- **You don't need to install CUDA Toolkit separately** - PyTorch includes the necessary CUDA runtime libraries
- **Driver version matters** - Your NVIDIA driver must support the CUDA version you're installing
- **CPU fallback** - If CUDA isn't available, the code will automatically use CPU (as it does now)

## Quick Reference

Check PyTorch CUDA support:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

Check GPU info:

```powershell
nvidia-smi
```

List available CUDA versions for PyTorch:
Visit: <https://pytorch.org/get-started/locally/>
