# Face-Attendance-System-Computer-Vision

## Overview
The Face Recognition Attendance System is an AI-powered desktop application designed to automate attendance recording using facial recognition. Traditional attendance methods are often manual, time-consuming, and prone to errors or proxy attendance. This system leverages computer vision and machine learning to recognize individuals from a live video stream and log attendance in real time.

## Methodology
The system operates in three main stages:
1. Dataset preparation and model usage
2. Model serialization and loading
3. Real-time inference through a graphical user interface (GUI)

Two face recognition pipelines were developed and compared:

- **InsightFace + Logistic Regression:** Uses pretrained InsightFace embeddings with a logistic regression classifier.
- **MTCNN + FaceNet + Cosine Similarity:** Uses MTCNN for face detection and FaceNet embeddings with cosine similarity for recognition.

### 1. Dataset Collection and Preparation
- Custom dataset with 14 students from the university.
- Each student provided 5–6 high-resolution images with varied poses, lighting, and expressions.
- Images were organized in folders named after each student, totaling approximately 80–85 images.


### 2. Multi-Model Approach Overview
The two pipelines allow evaluation of:

- Embedding quality and separability
- Classification vs. metric-based recognition
- Real-time feasibility and accuracy
- Ease of adding new identities without retraining

---

## InsightFace + Logistic Regression Pipeline

### 2.1 Face Embedding Extraction
- Used the `buffalo_l` model from InsightFace to extract 512-dimensional embeddings.
- Detected faces from dataset images, selecting the largest face when multiple were found.
- Stored embeddings with corresponding identity labels.

### 2.2 Model Training and Serialization
- Trained a logistic regression classifier (scikit-learn) on the embeddings.
- Label encoded student names into numeric labels.
- Serialized the classifier and label encoder for reuse.

### 2.3 Real-Time Recognition and Attendance Logging
- Developed a desktop GUI using `tkinter` and `ttkbootstrap`.
- Webcam feed is processed in real time.
- Detected faces are embedded and classified; attendance is logged with timestamp.
- Confidence threshold applied to filter uncertain predictions.
- Attendance logs can be exported as CSV files.

---

## FaceNet + Cosine Similarity Pipeline (with MTCNN Detection)

### 3.1 Face Detection and Alignment
- Used `keras-facenet` integrating MTCNN for detection and alignment.
- Faces detected with a confidence threshold of 0.95.
- Faces aligned before embedding extraction.

### 3.2 Data Normalization and Encoding
- L2 normalization applied to embeddings.
- Labels encoded with `LabelEncoder`.
- Embeddings and encoders serialized for inference.

### 3.3 Visual Validation
- Used t-SNE to reduce embeddings to 2D for visualization.
- Demonstrated clear separation between individual identities.

### 3.4 Recognition via Cosine Similarity
- Compared query embeddings with stored embeddings using cosine similarity.
- Threshold of 0.7 used for recognition.
- No retraining needed when adding new identities.

### 3.5 Real-Time Webcam Recognition
- Webcam frames processed live.
- Detected faces annotated with bounding boxes and predicted names.
- Recognition session controlled via GUI (`q` to quit).

---

## Results and Evaluation
- Tested both pipelines on a dataset with known and unknown faces.
- **InsightFace + Logistic Regression** showed higher confidence and accuracy.
- **FaceNet + Cosine Similarity** had lower confidence and missed some known faces.
- InsightFace pipeline was more robust to lighting and pose variations.

---

## Ethical Implications

### Privacy and Consent
- All participants voluntarily consented for academic use.
- No public or internet images used.
- Real-world deployment requires explicit, revocable consent and transparency.

### Data Security
- Embeddings stored locally with restricted access.
- Future recommendations include encryption and role-based access.

### Fairness and Bias
- Dataset limited in diversity, which limits generalization.
- Ethical deployment requires bias auditing and diverse datasets.

### Surveillance Risks
- System designed for attendance only, not continuous monitoring.
- Institutions should implement clear policies on system use.

### Transparency and Accountability
- Users should have access to logs and means to report errors.
- Institutions must monitor and review system behavior and fairness.

---

## Usage
1. Prepare your dataset in the specified folder structure.
2. Train or load pretrained models for InsightFace + Logistic Regression or FaceNet + Cosine Similarity.
3. Launch the GUI application to start real-time attendance logging.
4. Export attendance logs as CSV.

---

## Technologies Used
- Python 3
- OpenCV
- InsightFace (`buffalo_l` model)
- Keras-FaceNet and MTCNN
- Scikit-learn (Logistic Regression, Label Encoding, Normalization)
- Tkinter and ttkbootstrap for GUI

---

## License
This project is for academic purposes. Please obtain consent and follow ethical guidelines when deploying.

---

*For detailed implementation and source code, please refer to the project repository.*


