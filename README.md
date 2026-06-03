# Face Identity Validation

This repository provides a lightweight face enrollment and recognition system using [InsightFace](https://github.com/deepinsight/insightface) and ONNX Runtime. It allows you to enroll faces either live via a webcam or statically from existing passport-style photos, and then perform live real-time face recognition.

## Features
- **Live Face Enrollment (`enroll.py`)**: Capture multiple face samples (front, sides, up, down) using a webcam and save all embeddings to robustly handle occlusions and angles.
- **Batch Photo Enrollment (`enroll_from_photos.py`)**: Bulk-enroll identities from a directory of photos (e.g. `john.jpg`, `alice.png`).
- **Live Recognition (`recognise.py`)**: Real-time webcam inference to detect and recognize faces against the enrolled database.
- **Hardware Acceleration**: Built-in support for TensorRT and CUDA execution providers for high-performance inference via ONNX Runtime.

## Requirements

You can install the dependencies via the provided `requirements.txt`:
```bash
pip install -r requirements.txt
```

Key dependencies:
- `insightface`
- `onnxruntime-gpu` (Recommended for CUDA/TensorRT acceleration)
- `opencv-python`
- `numpy`

## Usage

### 1. Enrolling Faces from Photos
If you have existing photos of individuals you'd like to add to the database, place them in a directory. Ensure the filenames reflect the person's name (e.g., `alice.jpg`, `bob.png`).

```bash
python face/enroll_from_photos.py /path/to/photos
```
Optional arguments:
- `--database /path/to/custom_db`: Override the default `database/` location.
- `--force`: Overwrite existing entries in the database.

### 2. Live Enrollment
To enroll a face live using your webcam:
```bash
python face/enroll.py
```
You will be prompted to enter the person's name. The system will collect multiple face samples and save the mean embedding to the `database` directory. Press `Q` to cancel.

### 3. Live Recognition
To start real-time face recognition using your webcam:
```bash
python face/recognise.py
```
The script will load the known identities from the `database/` directory and overlay the recognized names and confidence scores on the webcam feed. Press `Q` to exit.

## Architecture & Internals
- **Embeddings Storage**: Identities are stored as `.npy` embedding files in the `face/database/` directory. They are saved as 2D arrays (matrices) to support multi-angle matching.
- **InsightFace Model**: Uses the `buffalo_l` (ResNet50) model out of the box with `det_size=(320, 320)`. This larger model offers much greater resilience to occlusions and profile angles.
- **Inference Providers**: Prioritizes `TensorrtExecutionProvider` and `CUDAExecutionProvider` falling back to `CPUExecutionProvider` if GPUs are unavailable. TensorRT engines are cached automatically in `face/trt_cache/` to speed up subsequent loads.
