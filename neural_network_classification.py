import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import pickle
from sklearn.decomposition import PCA
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical

classes = ['Pout' ,'Open_Mouth',"Neutral", 'Eyes_Close', 'Smile', 'Inward_Lip_Press', 'Show_Teeth']
# Load the dataset
file_path = "/content/drive/MyDrive/SIH2024/LivenessDetection/FinalData/jaf_angle_feature150.xlsx"
file_path ="custom_xangle_feature150.xlsx"
data = pd.read_excel(file_path, engine='openpyxl')
# Mapping dictionary for replacement
label_mapping = {
    "FE": 0,
    "SA": 1,
    "HA": 2,
    "NE": 3,
    "AN": 4,
    "SU": 5,
    "DI": 6
}

# Replace values in the Action_Label column
data['Action_Label'] = data['Action_Label'].replace(label_mapping)

# Extract class labels

class_counts = data['Action_Label'].value_counts()
print(class_counts)
# Filter the dataset for included classes
filtered_data = data[data['Action_Label'].isin(classes)]
#filtered_data=data
# Separate features and labels
X = filtered_data.drop(columns=['Action_Label', 'Image_Name']).values  # Features
y = filtered_data['Action_Label'].values  # Labels
classes = filtered_data['Action_Label'].unique()
print(classes)
# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA (Optional)
pca = PCA(n_components=0.95)  # Retain 95% variance
X_selected = pca.fit_transform(X_scaled)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_one_hot, test_size=0.20, random_state=42, stratify=y_one_hot
)

# Build the neural network
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.4),

    Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(y_train.shape[1], activation='softmax')
])

# Compile the model
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Save the best model during training
best_model_path = "Custom_xangle/best_model.h5"
checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_accuracy", verbose=1)

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=32, callbacks=[checkpoint])

# Save model, scaler, and label encoder using pickle
save_path = "Custom_xangle/model_with_scaler_and_encoder.pkl"
with open(save_path, 'wb') as file:
    pickle.dump({'model_path': best_model_path, 'scaler': scaler, 'label_encoder': label_encoder, 'pca': pca}, file)

print(f"Model, scaler, and label encoder saved to {save_path}")

# Load saved components
with open(save_path, 'rb') as file:
    loaded_objects = pickle.load(file)

loaded_model = load_model(loaded_objects['model_path'])
loaded_scaler = loaded_objects['scaler']
loaded_label_encoder = loaded_objects['label_encoder']
loaded_pca = loaded_objects['pca']

# Evaluate the best model
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f"Loaded Model Test Accuracy: {accuracy * 100:.2f}%")

# Predict and generate the report
y_pred_one_hot = loaded_model.predict(X_test)
y_pred = np.argmax(y_pred_one_hot, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=loaded_label_encoder.classes_))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=loaded_label_encoder.classes_,
            yticklabels=loaded_label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
