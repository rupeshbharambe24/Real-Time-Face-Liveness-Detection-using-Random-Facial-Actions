# 🎭 Real-Time Face Liveness Detection Using Random Facial Actions

This project implements a **face liveness detection system** that challenges a user with random facial actions like smiling, blinking, or showing teeth, and verifies whether those actions are performed correctly. This helps in distinguishing **live users** from **spoofing attempts** using photos or videos.

---

## 📌 Features

- Landmark detection using **MediaPipe**
- Normalization and feature extraction via **cosine angles**
- Classification with a **neural network**
- **10-fold cross-validation** for performance evaluation
- **Live camera** testing with randomized action prompts

---

## 📂 Project Pipeline (Execution Order)

Run each file **in the sequence below** to complete the liveness detection workflow:

| Step | File | Description |
|------|------|-------------|
| 1️⃣ | `landmarks_extraction_and_normalization.py` | Extracts and normalizes facial landmarks from dataset images |
| 2️⃣ | `cosine_Angle_Feature_Extraction_for_All_landmarks.py` | Computes cosine angle features from all landmark combinations |
| 3️⃣ | `Cosine_Angle_X_axis_Feature_Extraction_for_All_landmarks.py` | Computes cosine angles projected along the X-axis |
| 4️⃣ | `neural_network_classification.py` | Trains a neural network to classify facial actions |
| 5️⃣ | `10_fold_classification.py` | Performs 10-fold cross-validation for performance metrics |
| ✅ | `testing_on_single_sample.py` | Predicts facial action class for a single sample |
| 🎥 | `live_camera_testing.py` | Tests live webcam input against random facial action prompts |

---

## 🛠️ Requirements

Make sure Python 3.7+ is installed.

Install all dependencies using pip:

```bash
pip install mediapipe opencv-python numpy pandas scikit-learn tensorflow openpyxl
````

---

## ▶️ How to Run

1. **Landmark Extraction**

   ```bash
   python landmarks_extraction_and_normalization.py
   ```

2. **Feature Extraction**

   ```bash
   python cosine_Angle_Feature_Extraction_for_All_landmarks.py
   python Cosine_Angle_X_axis_Feature_Extraction_for_All_landmarks.py
   ```

3. **Model Training**

   ```bash
   python neural_network_classification.py
   ```

4. **Cross-Validation**

   ```bash
   python 10_fold_classification.py
   ```

5. **Single Sample Test**

   ```bash
   python testing_on_single_sample.py
   ```

6. **Live Camera Liveness Check**

   ```bash
   python live_camera_testing.py
   ```

---

## 📁 Directory Structure

```
├── NewDataset/                         # Contains subfolders for each facial action class
│   ├── Smile/
│   ├── Pout/
│   └── ...
├── landmarks_extraction_and_normalization.py
├── cosine_Angle_Feature_Extraction_for_All_landmarks.py
├── Cosine_Angle_X_axis_Feature_Extraction_for_All_landmarks.py
├── neural_network_classification.py
├── 10_fold_classification.py
├── testing_on_single_sample.py
├── live_camera_testing.py
├── normalized_landmarks_xy.xlsx       # Output of landmark normalization
├── cosine_angle_features.csv          # Output of cosine angle features
├── cosine_x_axis_features.csv         # Output of X-axis projected features
└── model/                             # Saved trained model (optional)
```

---

## ✅ Facial Actions Used

* Smile 😄
* Pout 😗
* Show Teeth 🦷
* Open Mouth 🗣️
* Close Eyes 😑
* Inward Lip Press 😬
  (*You can add more classes by updating the dataset and retraining.*)

---

## 📈 Output

* **Accuracy Report** from 10-fold validation
* **Prediction** for each random action
* **Real-time validation** using your webcam

---

## 🧪 Example Use Cases

* **Anti-spoofing** in facial recognition login systems
* **Secure user verification** in mobile or desktop apps
* **Liveness check** for online proctoring platforms

---

## 📝 License

This project is licensed under the **MIT License**.
Developed by [Rupesh Bharambe](https://github.com/rupeshbharambe24)

---

## 🤝 Contributions

Feel free to raise issues, suggest enhancements, or open a pull request.

---

## 📬 Contact

For questions or collaborations, reach out via GitHub or email listed on your profile.

```

