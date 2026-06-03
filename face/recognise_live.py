import cv2
import numpy as np
import os

from insightface.app import FaceAnalysis

# ------------------------------
# CONFIG
# ------------------------------

DATABASE_DIR = os.path.join(
    os.path.dirname(__file__), "database"
)
THRESHOLD = 0.45

# ------------------------------
# SIMILARITY
# ------------------------------

def cosine_similarity(a, b):
    # a: (512,)
    # b: (N, 512) or (512,)
    if b.ndim == 1:
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    else:
        # matrix multiplication to get similarity with all angles
        sims = np.dot(b, a) / (np.linalg.norm(b, axis=1) * np.linalg.norm(a))
        return np.max(sims)

# ------------------------------
# FACE RECOGNIZER CLASS
# ------------------------------

class FaceRecognizer:
    """
    Wraps InsightFace for on-demand face recognition.
    Designed to be instantiated once and reused across
    frames in vision_runtime.py.
    """

    def __init__(self, providers=None):

        if providers is None:
            trt_cache_path = os.path.join(os.path.dirname(__file__), "trt_cache")
            os.makedirs(trt_cache_path, exist_ok=True)
            providers = [
                ("TensorrtExecutionProvider", {
                    "device_id": 0,
                    "trt_max_workspace_size": 1073741824,  # 1GB
                    "trt_fp16_enable": True,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": trt_cache_path
                }),
                "CUDAExecutionProvider",
                "CPUExecutionProvider"
            ]

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=providers
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )

        self.database = {}
        self._load_database()

    # --------------------------
    # DATABASE
    # --------------------------

    def _load_database(self):

        self.database = {}

        if not os.path.isdir(DATABASE_DIR):
            print(
                f"[WARN] Database dir not found: "
                f"{DATABASE_DIR}"
            )
            return

        for file in os.listdir(DATABASE_DIR):

            if file.endswith(".npy"):

                name = file[:-4]

                self.database[name] = np.load(
                    os.path.join(DATABASE_DIR, file)
                )

        print("\nLoaded identities:")

        for person in self.database:
            print(f"  {person}")

    def reload_database(self):
        """Re-scan the database directory."""
        self._load_database()

    # --------------------------
    # IDENTIFY
    # --------------------------

    def _identify(self, embedding):

        best_name = "Unknown"
        best_score = -1

        for name, db_emb in self.database.items():

            score = cosine_similarity(
                embedding,
                db_emb
            )

            if score > best_score:
                best_score = score
                best_name = name

        if best_score < THRESHOLD:
            return "Unknown", best_score

        return best_name, best_score

    # --------------------------
    # PUBLIC API
    # --------------------------

    def recognize_person(self, frame, box):
        """
        Detect and recognize a face within the given
        bounding box region of the frame.

        Args:
            frame: Full BGR frame from the camera.
            box: [x1, y1, x2, y2] bounding box of the
                 person detection.

        Returns:
            (name, score) tuple.

        Raises:
            RuntimeError: If no face is found in the
                          cropped region.
        """
        x1, y1, x2, y2 = box

        # Clamp to frame boundaries
        h, w = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            raise RuntimeError("Empty crop region")

        faces = self.app.get(crop)

        if not faces:
            raise RuntimeError(
                "No face detected in crop"
            )

        # Use the largest detected face
        best_face = max(
            faces,
            key=lambda f: (
                (f.bbox[2] - f.bbox[0])
                * (f.bbox[3] - f.bbox[1])
            )
        )

        return self._identify(best_face.embedding)


# ------------------------------
# STANDALONE DEMO
# ------------------------------

if __name__ == "__main__":

    recognizer = FaceRecognizer()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        faces = recognizer.app.get(frame)

        for face in faces:

            bx1, by1, bx2, by2 = map(
                int,
                face.bbox
            )

            name, score = recognizer._identify(
                face.embedding
            )

            cv2.rectangle(
                frame,
                (bx1, by1),
                (bx2, by2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} {score:.2f}",
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()