import cv2
import numpy as np
import os

# Force OpenCV to use TCP for RTSP to prevent timeout drops
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

from insightface.app import FaceAnalysis

# ------------------------------
# CONFIG
# ------------------------------

DATABASE_DIR = os.path.join(
    os.path.dirname(__file__), "database"
)
THRESHOLD = 0.45
CAMERA_STREAM = "rtsp://172.30.0.102"
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
            # Temporarily restricted to CPU to hide the massive missing DLL errors.
            # Once CUDA 12 is installed, we can add "CUDAExecutionProvider" back.
            providers = ["CPUExecutionProvider"]

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

def open_rtsp_stream(rtsp_url: str):
    """
    Opens an RTSP stream with authentication and displays frames.
    :param rtsp_url: Full RTSP URL including username and password.
    """
    try:
        # Open the RTSP stream
        cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)

        if not cap.isOpened():
            raise ConnectionError("Failed to connect to RTSP stream. Check URL and credentials.")

        print("Connected to RTSP stream. Press 'q' to quit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from stream.")
                break

            cv2.imshow("RTSP Stream", frame)

            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

# ------------------------------
# STANDALONE DEMO
# ------------------------------

import threading

class ThreadedCamera:
    """
    A threaded camera class that constantly reads frames from the RTSP stream
    in the background. This prevents the OpenCV buffer from filling up and 
    causing massive lag when the CPU face recognition is running slowly.
    """
    def __init__(self, src=0):
        self.capture = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.status = False
        self.frame = None
        
        if self.capture.isOpened():
            self.status, self.frame = self.capture.read()
            self.thread = threading.Thread(target=self.update, args=())
            self.thread.daemon = True
            self.thread.start()
        else:
            print("[ERROR] Could not connect to the RTSP stream!")

    def update(self):
        while self.status:
            self.status, self.frame = self.capture.read()

    def read(self):
        return self.status, self.frame

    def release(self):
        self.status = False
        self.capture.release()

if __name__ == "__main__":
    from tracker import CentroidTracker
    username = "admin"
    password = "Vertiv@123"
    ip = "172.30.0.102"
    port = 554
    # Standard CP Plus / Dahua RTSP path
    path = "cam/realmonitor?channel=1&subtype=0"
    
    recognizer = FaceRecognizer()
    rtsp_url = f"rtsp://{username}:{password}@{ip}:{port}/{path}"
    
    print(f"[INFO] Connecting to {ip}...")
    cap = ThreadedCamera(rtsp_url)
    
    ct = CentroidTracker(max_disappeared=30, max_distance=100)
    track_identities = {}

    while cap.status:
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        faces = recognizer.app.get(frame)
        
        rects = []
        face_info = {}

        for face in faces:
            bx1, by1, bx2, by2 = map(int, face.bbox)
            rect = (bx1, by1, bx2, by2)
            rects.append(rect)
            
            name, score = recognizer._identify(face.embedding)
            face_info[rect] = (name, score)

        objects = ct.update(rects)

        for object_id, (centroid, rect) in objects.items():
            raw_name = "Unknown"
            raw_score = 0.0
            
            if rect in face_info:
                raw_name, raw_score = face_info[rect]
            
            if raw_name != "Unknown":
                if object_id not in track_identities or raw_score > track_identities[object_id][1]:
                    track_identities[object_id] = (raw_name, raw_score)

            final_name, final_score = track_identities.get(object_id, ("Unknown", raw_score))

            bx1, by1, bx2, by2 = rect
            
            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            label = f"ID {object_id}: {final_name} {final_score:.2f}"
            cv2.putText(
                frame,
                label,
                (bx1, by1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

        cv2.imshow("Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()