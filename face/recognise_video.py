import cv2
import argparse
import time
import sys
from pathlib import Path

# Import the existing FaceRecognizer class
from recognise_live import FaceRecognizer
from tracker import CentroidTracker

def process_video(input_path, output_path=None, display=True):
    print("[INFO] Loading Face Recognizer Model...")
    recognizer = FaceRecognizer()
    
    print("[INFO] Initializing Face Tracker...")
    ct = CentroidTracker(max_disappeared=30, max_distance=100)
    track_identities = {} # Maps object_id -> (name, max_score)
    
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video file: {input_path}")
        sys.exit(1)

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = None
    if output_path:
        # Defaulting to mp4v codec for mp4 files
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"[INFO] Writing output to {output_path}")

    print(f"[INFO] Processing video: {input_path}")
    print(f"       Resolution: {width}x{height}, FPS: {fps:.2f}, Frames: {total_frames}")

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Use FaceRecognizer app to get faces
        faces = recognizer.app.get(frame)

        rects = []
        face_info = {}

        for face in faces:
            # Get bounding box
            bx1, by1, bx2, by2 = map(int, face.bbox)
            rect = (bx1, by1, bx2, by2)
            rects.append(rect)
            
            # Raw identification from embedding
            name, score = recognizer._identify(face.embedding)
            face_info[rect] = (name, score)

        # Update tracker with current faces
        objects = ct.update(rects)

        for object_id, (centroid, rect) in objects.items():
            raw_name = "Unknown"
            raw_score = 0.0
            
            # If the tracked object corresponds to a face detected in *this* frame
            if rect in face_info:
                raw_name, raw_score = face_info[rect]
            
            # If the raw recognition is highly confident, lock the identity for this track
            if raw_name != "Unknown":
                # Only update if we didn't know who they were, or if the new score is better
                if object_id not in track_identities or raw_score > track_identities[object_id][1]:
                    track_identities[object_id] = (raw_name, raw_score)

            # Retrieve the locked identity (if any)
            final_name, final_score = track_identities.get(object_id, ("Unknown", raw_score))

            bx1, by1, bx2, by2 = rect
            
            # Draw bounding box
            color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, 2)

            # Draw label with Track ID
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

        if out:
            out.write(frame)

        if display:
            cv2.imshow("Video Face Recognition", frame)
            # Press 'q' to quit early
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Interrupted by user.")
                break

    end_time = time.time()
    elapsed = end_time - start_time
    
    # Avoid division by zero if elapsed is too small
    actual_fps = frame_count / elapsed if elapsed > 0 else 0
    
    print(f"\n[INFO] Finished processing.")
    print(f"[INFO] Processed {frame_count} frames in {elapsed:.2f} seconds ({actual_fps:.2f} FPS).")

    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recognize faces in a video file.")
    parser.add_argument("input_video", help="Path to the input video file")
    parser.add_argument("--output", "-o", help="Path to save the output video (e.g., output.mp4)", default=None)
    parser.add_argument("--no-display", action="store_true", help="Disable video display during processing (useful for fast background processing)")

    args = parser.parse_args()

    process_video(
        input_path=args.input_video,
        output_path=args.output,
        display=not args.no_display
    )
