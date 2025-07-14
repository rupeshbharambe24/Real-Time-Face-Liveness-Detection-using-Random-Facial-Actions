import os
import cv2
import mediapipe as mp
import pandas as pd

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Dataset folder path
dataset_path = r"NewDataset"
output_file = "normalized_landmarks_xy.xlsx"

# Initialize variables
output_data = []
batch_size = 100  # Save data after every 100 images
classes = ['Pout', 'Open_Mouth', 'Eyes_Close', 'Smile', 'Inward_Lip_Press', 'Show_Teeth']
classes=["Brow_Raise","Surprise","Nose_Wrinkle","Cheek_Puffing"]
# Key landmarks for height and width normalization
HEIGHT_LANDMARKS = (10, 152)  # Chin and top of the head
WIDTH_LANDMARKS = (93, 323)  # Left cheek and right cheek

# Helper function to write data to Excel
def save_to_excel(data, file_path, columns, mode='a'):
    """Save data to Excel file with specified mode"""
    df = pd.DataFrame(data, columns=columns)
    
    if mode == 'w' or not os.path.exists(file_path):
        # Create new file or overwrite existing
        df.to_excel(file_path, index=False, engine='openpyxl')
    else:
        # Append to existing file
        with pd.ExcelWriter(file_path, mode='a', engine='openpyxl', if_sheet_exists='overlay') as writer:
            start_row = writer.sheets['Sheet1'].max_row
            df.to_excel(writer, index=False, header=False, startrow=start_row)


# Process each subfolder and images
for pose_folder in os.listdir(dataset_path):
    print("Processing folder:", pose_folder)
    cnt = 0

    pose_path = os.path.join(dataset_path, pose_folder)
    if os.path.isdir(pose_path):
        pose_label = pose_folder  # e.g., "Pout", "Smile", etc.

        for image_file in os.listdir(pose_path):
            if cnt > 3000:  # Limit to 500 images per class
                break
            cnt += 1
            print(f"Processing: {image_file}")
            image_path = os.path.join(pose_path, image_file)
            if image_file.endswith(('.png', '.jpg', '.jpeg')):
                # Read the image
                image = cv2.imread(image_path)
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Process the image with FaceMesh
                results = face_mesh.process(rgb_image)
                if results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        # Extract x, y coordinates for all landmarks
                        landmarks = [(lm.x, lm.y) for lm in face_landmarks.landmark]

                        # Calculate face bounding box dimensions
                        w1 = landmarks[WIDTH_LANDMARKS[0]][0]
                        w2 = landmarks[WIDTH_LANDMARKS[1]][0]
                        h1 = landmarks[HEIGHT_LANDMARKS[0]][1]
                        h2 = landmarks[HEIGHT_LANDMARKS[1]][1]
                        total_height = abs(h2 - h1)
                        total_width = abs(w2 - w1)

                        # Normalize all landmark coordinates relative to the face size
                        normalized_landmarks = [
                            ((lm[0] - w1) / total_width, (lm[1] - h1) / total_height) for lm in landmarks
                        ]

                        # Flatten normalized landmarks into a single list
                        normalized_landmarks_flat = [coord for lm in normalized_landmarks for coord in lm]

                        # Append pose label, image name, and normalized landmarks
                        output_data.append([pose_label, image_file] + normalized_landmarks_flat)

                        # Save periodically to avoid memory issues
                        if len(output_data) >= batch_size:
                            num_landmarks = len(output_data[0]) - 2  # Exclude 'Pose' and 'Image'
                            columns = ["Pose", "Image"] + [f"L{i}" for i in range(num_landmarks)]
                            save_to_excel(output_data, output_file, columns, mode='a')
                            output_data = []  # Clear the list after saving

# Save any remaining data
if len(output_data) >= batch_size:
    num_landmarks = len(output_data[0]) - 2
    columns = ["Pose", "Image"] + [f"L{i}" for i in range(num_landmarks)]
    save_to_excel(output_data, output_file, columns)
    output_data = []
    print("Final batch saved.")

print("Processing complete.")