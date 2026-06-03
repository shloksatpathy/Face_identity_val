"""
Enroll faces into the known_database from passport
photos (or any static face images).

Usage:
    python enroll_from_photos.py <photos_dir>
Supported formats: .jpg .jpeg .png .bmp .webp
"""

import cv2
import numpy as np
import os
import sys
import argparse
from pathlib import Path

from insightface.app import FaceAnalysis

# --------------------------------
# CONFIG
# --------------------------------

DATABASE_DIR = os.path.join(
    os.path.dirname(__file__), "database"
)

SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".webp"
}

MIN_FACE_SIZE = 60  # Smaller than live enrollment
                     # since passport photos are
                     # typically tight crops

# --------------------------------
# MODEL
# --------------------------------

def _build_app():
    """
    Build the InsightFace app with the same
    provider chain used by enroll.py and
    recognise.py for embedding compatibility.
    """
    trt_cache_path = os.path.join(
        os.path.dirname(__file__), "trt_cache"
    )
    os.makedirs(trt_cache_path, exist_ok=True)

    providers = [
        ("TensorrtExecutionProvider", {
            "device_id": 0,
            "trt_max_workspace_size": 1073741824,
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": trt_cache_path
        }),
        "CUDAExecutionProvider",
        "CPUExecutionProvider"
    ]

    app = FaceAnalysis(
        name="buffalo_s",
        providers=providers
    )

    app.prepare(
        ctx_id=0,
        det_size=(320, 320)
    )

    return app


# --------------------------------
# ENROLLMENT FROM PHOTO
# --------------------------------

def enroll_single_photo(app, image_path, name,
                        database_dir, force=False):
    """
    Extract a face embedding from a single photo
    and save it to the database.

    Args:
        app: InsightFace FaceAnalysis instance.
        image_path: Path to the photo file.
        name: Identity name for this person.
        database_dir: Path to the database dir.
        force: Overwrite existing entries.

    Returns:
        True if successfully enrolled.
    """
    save_path = os.path.join(
        database_dir,
        f"{name.lower()}.npy"
    )

    # Check for existing entry
    if os.path.exists(save_path) and not force:
        print(
            f"  [SKIP] '{name}' already exists. "
            f"Use --force to overwrite."
        )
        return False

    # Read image
    img = cv2.imread(str(image_path))

    if img is None:
        print(
            f"  [ERROR] Cannot read image: "
            f"{image_path}"
        )
        return False

    # Detect faces
    faces = app.get(img)

    if not faces:
        print(
            f"  [ERROR] No face detected in: "
            f"{image_path}"
        )
        return False

    # Filter by minimum face size
    valid_faces = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        w = x2 - x1
        h = y2 - y1
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            valid_faces.append(face)

    if not valid_faces:
        print(
            f"  [ERROR] Face too small in: "
            f"{image_path} "
            f"(min {MIN_FACE_SIZE}px)"
        )
        return False

    if len(valid_faces) > 1:
        print(
            f"  [WARN] Multiple faces ({len(valid_faces)}) "
            f"in {image_path}. Using largest."
        )

    # Pick the largest face (most reliable)
    best_face = max(
        valid_faces,
        key=lambda f: (
            (f.bbox[2] - f.bbox[0])
            * (f.bbox[3] - f.bbox[1])
        )
    )

    # Save embedding
    np.save(save_path, best_face.embedding)

    score = best_face.det_score
    bbox = [int(c) for c in best_face.bbox]

    print(
        f"  [OK] Enrolled '{name}' "
        f"(det_score={score:.3f}, "
        f"bbox={bbox})"
    )

    return True


def enroll_from_directory(photos_dir,
                          database_dir=None,
                          force=False):
    """
    Batch enroll all photos in a directory.

    Args:
        photos_dir: Path to the photos directory.
        database_dir: Override database location.
        force: Overwrite existing entries.

    Returns:
        (enrolled, skipped, failed) counts.
    """
    photos_dir = Path(photos_dir)

    if database_dir is None:
        database_dir = DATABASE_DIR

    os.makedirs(database_dir, exist_ok=True)

    if not photos_dir.is_dir():
        print(f"[ERROR] Not a directory: {photos_dir}")
        return 0, 0, 0

    # Collect image files
    image_files = sorted([
        f for f in photos_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_files:
        print(
            f"[ERROR] No images found in: "
            f"{photos_dir}\n"
            f"Supported: "
            f"{', '.join(SUPPORTED_EXTENSIONS)}"
        )
        return 0, 0, 0

    print(f"\n{'='*50}")
    print(f"  PASSPORT PHOTO ENROLLMENT")
    print(f"{'='*50}")
    print(f"  Source : {photos_dir.resolve()}")
    print(f"  Database: {os.path.abspath(database_dir)}")
    print(f"  Photos : {len(image_files)}")
    print(f"  Force  : {force}")
    print(f"{'='*50}\n")

    # Initialize model
    print("[INFO] Loading InsightFace model...")
    app = _build_app()
    print("[INFO] Model ready.\n")

    enrolled = 0
    skipped = 0
    failed = 0

    for image_path in image_files:

        # Name from filename (without extension)
        name = image_path.stem

        print(f"Processing: {image_path.name}")

        result = enroll_single_photo(
            app, image_path, name,
            database_dir, force
        )

        if result:
            enrolled += 1
        elif os.path.exists(
            os.path.join(
                database_dir,
                f"{name.lower()}.npy"
            )
        ):
            skipped += 1
        else:
            failed += 1

    # Summary
    print(f"\n{'='*50}")
    print(f"  ENROLLMENT COMPLETE")
    print(f"{'='*50}")
    print(f"  Enrolled : {enrolled}")
    print(f"  Skipped  : {skipped}")
    print(f"  Failed   : {failed}")
    print(f"  Total DB : {len(list(Path(database_dir).glob('*.npy')))}")
    print(f"{'='*50}\n")

    return enrolled, skipped, failed


# --------------------------------
# CLI
# --------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Enroll faces from passport photos "
            "into the known database."
        )
    )

    parser.add_argument(
        "photos_dir",
        help=(
            "Directory containing passport photos. "
            "Filenames become identity names "
            "(e.g. john.jpg → 'john')."
        )
    )

    parser.add_argument(
        "--database", "-d",
        default=None,
        help=(
            "Override database directory "
            f"(default: {DATABASE_DIR})"
        )
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Overwrite existing database entries."
    )

    args = parser.parse_args()

    enroll_from_directory(
        args.photos_dir,
        database_dir=args.database,
        force=args.force
    )
