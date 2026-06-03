import cv2
from insightface.app import FaceAnalysis

app = FaceAnalysis(
    providers=['CPUExecutionProvider']
)

app.prepare(ctx_id=0)

img = cv2.imread("person.png")

faces = app.get(img)

for face in faces:
    print("Embedding shape:", face.embedding.shape)