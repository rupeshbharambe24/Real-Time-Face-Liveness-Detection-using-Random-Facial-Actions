import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib  # For loading scaler
import pickle
# Define landmark triplets for angle computation
triplets = [(39, 61, 91), (39, 0, 91), (0, 91, 181), (0, 17, 181), (291, 321, 308),
            (269, 0, 321), (0, 17, 405), (0, 314, 321), (269, 291, 0), (159, 33, 145),
            (159, 133, 145), (386, 362, 374), (386, 263, 374), (291, 436, 294),
            (61, 216, 64), (55, 168, 285), (151, 107, 8), (151, 336, 8), (55, 151, 285),
            (334, 300, 445), (105, 70, 225), (61, 64, 0), (291, 294, 0), (61, 4, 291),
            (296, 417, 382), (66, 189, 154), (61, 135, 207), (66, 33, 108), (296, 263, 337),
            (105, 10, 334), (0, 61, 17), (61, 0, 292), (48, 4, 289), (206, 4, 426),
            (61, 4, 292), (292, 280, 289), (292, 426, 289), (159, 130, 145),
            (46, 105, 107), (48, 122, 107)]

triplets = [(40, 61, 291), (40, 61, 91), (40, 61, 118), (291, 61, 91), (291, 61, 118), (91, 61, 118),
            (0, 40, 61), (40, 0, 270), (0, 270, 291), (321, 291, 347), (321, 291, 61), (321, 291, 270),
            (347, 291, 61), (347, 291, 270), (61, 291, 270), (80, 78, 88), (13, 80, 78), (80, 13, 14),
            (80, 13, 311), (14, 13, 311), (308, 311, 13), (402, 308, 311), (17, 91, 61), (321, 17, 91),
            (17, 321, 291), (78, 88, 14), (88, 14, 402), (88, 14, 13), (402, 14, 13), (308, 402, 14),
            (216, 206, 49), (50, 216, 212), (50, 216, 206), (212, 216, 206), (216, 212, 204), (200, 204, 212),
            (424, 200, 204), (436, 426, 279), (432, 436, 280), (432, 436, 426), (280, 436, 426), (424, 432, 436),
            (432, 424, 200), (147, 70, 63), (105, 63, 70), (66, 105, 63), (105, 66, 107), (9, 107, 66),
            (9, 107, 6), (66, 107, 6), (118, 117, 143), (128, 118, 50), (128, 118, 61), (128, 118, 117),
            (50, 118, 61), (50, 118, 117), (61, 118, 117), (216, 50, 118), (296, 336, 6), (336, 296, 334),
            (296, 334, 293), (300, 293, 334), (372, 300, 293), (346, 372, 300), (280, 347, 346), (280, 347, 291),
            (280, 347, 357), (346, 347, 291), (346, 347, 357), (291, 347, 357), (347, 346, 372), (347, 280, 436),
            (336, 6, 107), (336, 6, 4), (107, 6, 4), (145, 33, 159), (33, 159, 145), (33, 159, 133),
            (145, 159, 133), (145, 133, 159), (33, 145, 133), (33, 145, 159), (133, 145, 159), (386, 362, 374),
            (362, 386, 374), (362, 386, 263), (374, 386, 263), (386, 263, 374), (362, 374, 386), (362, 374, 263),
            (386, 374, 263), (472, 469, 470), (469, 470, 471), (469, 472, 471), (472, 471, 470), (475, 474, 477),#old
            (474, 475, 476), (474, 477, 476), (475, 476, 477), (107, 9, 108), (337, 108, 9),
            (39, 61, 91), (39, 0, 91), (0, 91, 181), (0, 17, 181), (291, 321, 308), (269, 0, 321),
                (0, 17, 405), (0, 314, 321), (269, 291, 0), (159, 33, 145), (159, 133, 145), (386, 362, 374),
                (386, 263, 374), (291, 436, 294), (61, 216, 64), (55, 168, 285), (151, 107, 8), (151, 336, 8),
                (55, 151, 285), (334, 300, 445), (105, 70, 225), (61, 64, 0), (291, 294, 0), (61, 4, 291),
                (296, 417, 382), (66, 189, 154), (61, 135, 207),(66,33,108),(296,263,337),(105,10,334),
                (0,61,17),(61,0,292),(48,4,289),(206,4,426),(61,4,292),(292,280,289),(292,426,289),(159,130,145),
                (46,105,107),(48,122,107)]
# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Landmark indices for normalization
HEIGHT_LANDMARKS = (10, 152)  # Chin and top of the head
WIDTH_LANDMARKS = (93, 323)   # Left cheek and right cheek

def extract_landmarks(image_path):
    """Extract normalized facial landmarks."""
    image = cv2.imread(image_path)
    # Get image dimensions
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)  # Find the center of the image

    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, 45, 1.0)

    # Rotate the image
    r_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    image=r_image

    if image is None:
        print("❌ Error: Could not load image.")
        return None

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks = [(lm.x, lm.y) for lm in face_landmarks]

        # Normalize landmarks using face bounding box
        w1, _ = landmarks[WIDTH_LANDMARKS[0]]
        w2, _ = landmarks[WIDTH_LANDMARKS[1]]
        _, h1 = landmarks[HEIGHT_LANDMARKS[0]]
        _, h2 = landmarks[HEIGHT_LANDMARKS[1]]

        total_width = abs(w2 - w1)
        total_height = abs(h2 - h1)

        if total_width == 0 or total_height == 0:
            print("⚠ Warning: Face bounding box has zero width/height.")
            return None

        normalized_landmarks = [
            ((lm[0] - w1) / total_width, (lm[1] - h1) / total_height) for lm in landmarks
        ]

        return [coord for lm in normalized_landmarks for coord in lm]

    print("⚠ No face detected in the image.")
    return None
import numpy as np

def compute_cosineSim_angles(landmarks):
    """
    Compute rotation-invariant angles for a single sample using cosine similarity.
    Assumes x and y coordinates are stored in consecutive positions.
    """
    angles = []

    for (a, b, c) in triplets:
        try:
            x1, y1 = landmarks[a * 2], landmarks[a * 2 + 1]
            x2, y2 = landmarks[b * 2], landmarks[b * 2 + 1]
            x3, y3 = landmarks[c * 2], landmarks[c * 2 + 1]

            # Compute vectors AB and BC
            AB = np.array([x1 - x2, y1 - y2])
            BC = np.array([x3 - x2, y3 - y2])

            # Compute cosine similarity
            dot_product = np.dot(AB, BC)
            norm_AB = np.linalg.norm(AB)
            norm_BC = np.linalg.norm(BC)

            if norm_AB == 0 or norm_BC == 0:
                theta = 0  # Avoid division by zero
            else:
                cos_theta = np.clip(dot_product / (norm_AB * norm_BC), -1.0, 1.0)
                theta = np.degrees(np.arccos(cos_theta))

            angles.append(theta)
        except IndexError:
            print(f"⚠ Warning: Landmark index out of range in triplet {a, b, c}")
            angles.append(None)

    return np.array(angles, dtype=np.float32)
def compute_cosine_angles(landmarks):
    """Compute cosine angles for given landmark coordinates."""
    angles = []
    for (a, b, c) in triplets:
        try:
            Ax, Ay = landmarks[a * 2], landmarks[a * 2 + 1]
            Bx, By = landmarks[b * 2], landmarks[b * 2 + 1]
            Cx, Cy = landmarks[c * 2], landmarks[c * 2 + 1]

            A, B, C = np.array([Ax, Ay]), np.array([Bx, By]), np.array([Cx, Cy])
            AB, BC = A - B, C - B

            angle_rad = np.arctan2(BC[1], BC[0]) - np.arctan2(AB[1], AB[0])
            angle_deg = np.degrees(angle_rad)

            if angle_deg < 0:
                angle_deg += 360
            angles.append(angle_deg)
        except IndexError:
            print(f"⚠ Warning: Landmark index out of range in triplet {a, b, c}")
            angles.append(None)

    return np.array(angles, dtype=np.float32)

def predict_expression(image_path, model, scaler, label_encoder):
    """Predict facial expression from a single image."""
    landmarks = extract_landmarks(image_path)

    if landmarks is None:
        print("❌ No valid landmarks extracted.")
        return

    angles = compute_cosineSim_angles(landmarks)
    features = angles

    # Ensure no NaN values in features
    if np.isnan(features).any():
        print("⚠ Warning: NaN values in feature vector. Skipping prediction.")
        return

    # Normalize features

    #features = np.unique(features)
    features = scaler.transform([features])
    # Make prediction
    pred_prob = model.predict(features)
    pred_class_index = np.argmax(pred_prob, axis=1)[0]
    predicted_label = label_encoder.inverse_transform([pred_class_index])[0]

    print(f"\n✅ Predicted Expression: {predicted_label}")

# Load model, scaler, label encoder, and PCA
load_path = "Custom_xangle/model_with_scaler_and_encoder.pkl"
with open(load_path, 'rb') as file:
    data = pickle.load(file)

# Extract the components
best_model_path = data['model_path']
scaler = data['scaler']
label_encoder = data['label_encoder']
best_model = load_model(best_model_path)  # Load the model properly

# Example Usage
image_path = r"img.jpg"
predict_expression(image_path, best_model, scaler, label_encoder)
