import cv2
import numpy as np
import os
import time

from insightface.app import FaceAnalysis

# --------------------------------
# CONFIG
# --------------------------------

DATABASE_DIR = os.path.join(
    os.path.dirname(__file__), "database"
)
NUM_SAMPLES = 30
MIN_FACE_SIZE = 120

os.makedirs(DATABASE_DIR, exist_ok=True)

# --------------------------------
# SHARED MODEL (lazy-loaded)
# --------------------------------

_app = None

def _get_app():
    global _app
    if _app is None:
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
        _app = FaceAnalysis(
            name="buffalo_l",
            providers=providers
        )
        _app.prepare(
            ctx_id=0,
            det_size=(320, 320)
        )
    return _app

# --------------------------------
# ENROLLMENT SESSION
# --------------------------------

class EnrollmentSession:
    """
    Non-blocking enrollment that processes one frame
    at a time from an existing camera loop.

    Usage in a main loop:
        session = EnrollmentSession("alice")
        ...
        frame = session.process_frame(frame)
        if session.is_complete():
            session.save()
    """

    def __init__(self, name, recognizer=None):

        self.name = name
        self.app = _get_app()
        self.recognizer = recognizer
        self.embeddings = []
        self.last_capture = 0
        self.done = False
        self.cancelled = False
        self._duplicate_checked = False

        print(f"\nCollecting {NUM_SAMPLES} samples...")
        print("Please follow the on-screen instructions to turn your head.")
        print("Press Q to cancel enrollment.\n")

    def process_frame(self, frame):
        """
        Analyse a single frame for face embeddings.
        Draws enrollment HUD overlay on the frame.
        Returns the annotated frame.
        """
        if self.done or self.cancelled:
            return frame

        faces = self.app.get(frame)

        for face in faces:

            x1, y1, x2, y2 = map(int, face.bbox)

            width = x2 - x1
            height = y2 - y1

            # Draw face box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Ignore tiny faces
            if width < MIN_FACE_SIZE or height < MIN_FACE_SIZE:
                continue

            # ----- DUPLICATE CHECK -----
            # On the first valid face, verify the
            # person isn't already in the database.
            if (
                not self._duplicate_checked
                and self.recognizer is not None
                and self.recognizer.database
            ):
                self._duplicate_checked = True

                match_name, match_score = (
                    self.recognizer._identify(
                        face.embedding
                    )
                )

                if match_name != "Unknown":
                    print(
                        f"[BLOCKED] Face already "
                        f"enrolled as '{match_name}' "
                        f"(score: {match_score:.2f}). "
                        f"Enrollment cancelled."
                    )
                    self.cancelled = True

                    # Show rejection on frame
                    cv2.putText(
                        frame,
                        f"ALREADY ENROLLED: "
                        f"{match_name}",
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2
                    )
                    return frame

            current_time = time.time()

            # Capture every 0.3 sec
            if current_time - self.last_capture > 0.3:

                self.embeddings.append(face.embedding)

                self.last_capture = current_time

                print(
                    f"Captured "
                    f"{len(self.embeddings)}/{NUM_SAMPLES}"
                )

        # Determine prompt based on progress
        progress = len(self.embeddings)
        if progress < NUM_SAMPLES * 0.2:
            instruction = "Look Straight"
        elif progress < NUM_SAMPLES * 0.4:
            instruction = "Turn Head Left"
        elif progress < NUM_SAMPLES * 0.6:
            instruction = "Turn Head Right"
        elif progress < NUM_SAMPLES * 0.8:
            instruction = "Look Up slightly"
        else:
            instruction = "Look Down slightly"

        # Draw enrollment HUD
        cv2.putText(
            frame,
            f"ENROLLING: {self.name} "
            f"({len(self.embeddings)}/{NUM_SAMPLES})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        # Draw instruction
        cv2.putText(
            frame,
            f"Action: {instruction}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        if len(self.embeddings) >= NUM_SAMPLES:
            self.done = True

        return frame

    def cancel(self):
        """Cancel the enrollment session."""
        self.cancelled = True
        print("[INFO] Enrollment cancelled")

    def is_complete(self):
        """True when enough samples collected."""
        return self.done

    def is_active(self):
        """True while still collecting."""
        return not self.done and not self.cancelled

    def save(self):
        """
        Save the mean embedding to the database.
        Call only after is_complete() returns True.
        """
        if len(self.embeddings) == 0:
            print("[ERROR] No samples collected")
            return False

        # Save all embeddings as (N, 512) for multi-pose recognition
        all_embeddings = np.array(self.embeddings)

        save_path = os.path.join(
            DATABASE_DIR,
            f"{self.name.lower()}.npy"
        )

        np.save(
            save_path,
            all_embeddings
        )

        print(f"\nSaved profile: {save_path}")
        print(f"Samples used: {len(self.embeddings)}")
        return True


# --------------------------------
# LEGACY WRAPPER (standalone use)
# --------------------------------

def enroll_person(name, src=0):
    """
    Blocking enrollment using its own camera.
    Use only when running enroll.py as a standalone
    script. For integration with vision_runtime.py,
    use EnrollmentSession instead.
    """
    session = EnrollmentSession(name)

    cap = cv2.VideoCapture(src)

    if not cap.isOpened():
        raise RuntimeError("Camera not opened")

    while session.is_active():

        ret, frame = cap.read()

        if not ret:
            break

        frame = session.process_frame(frame)

        cv2.imshow("Enrollment", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            session.cancel()
            break

    cap.release()
    cv2.destroyWindow("Enrollment")

    if session.is_complete():
        session.save()


# --------------------------------
# STANDALONE SCRIPT
# --------------------------------

if __name__ == "__main__":

    person_name = input(
        "Enter person name: "
    ).strip()

    if not person_name:
        raise ValueError("Invalid name")

    enroll_person(person_name)