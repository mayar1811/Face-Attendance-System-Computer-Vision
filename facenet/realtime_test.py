import cv2
import numpy as np
import pickle
from keras_facenet import FaceNet
from sklearn.metrics.pairwise import cosine_similarity

embedder = FaceNet()
print("✅ FaceNet model loaded")

data = np.load('embeddings_dataset.npz')
X_ref, y_ref = data['arr_0'], data['arr_1']

with open('label_encoder.pkl', 'rb') as f:
    out_encoder = pickle.load(f)

def cosine_match(query_emb, X_ref, y_ref, encoder, threshold=0.8):
    sims = cosine_similarity([query_emb], X_ref)[0]
    best_idx = np.argmax(sims)
    best_sim = sims[best_idx]
    label = encoder.inverse_transform([y_ref[best_idx]])[0]
    if best_sim >= threshold:
        return label, best_sim
    return "Unknown", best_sim

cap = cv2.VideoCapture(0)
frame_skip = 2
frame_count = 0

print("📷 Starting live face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_skip != 0:
        frame_count += 1
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = embedder.extract(rgb_frame, threshold=0.95)

    for face in results:
        x, y, w, h = face['box']
        emb = face['embedding']
        name, confidence = cosine_match(emb, X_ref, y_ref, out_encoder, threshold=0.7)

        label = f"{name} ({confidence:.2f})"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 0), 2)

    cv2.imshow("Live Face Recognition (Cosine)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
