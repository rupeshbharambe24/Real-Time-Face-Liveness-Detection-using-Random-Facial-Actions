# Stratified k-fold cross-validation of the MLP (10 folds)
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
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
print("import completed")

# Define the classes
classes = ['Pout', 'Open_Mouth', "Neutral", 'Eyes_Close', 'Smile', 'Inward_Lip_Press', 'Show_Teeth']
file_path = "data/angle_features_cosine.xlsx"  # output of src/angle_features_cosine.py
N_FOLDS = 10
os.makedirs("models/cv", exist_ok=True)
data = pd.read_excel(file_path, engine='openpyxl')

# Mapping dictionary for replacement
label_mapping = {
    "FE": '0',
    "SA": '1',
    "HA": '2',
    "NE": '3',
    "AN": '4',
    "SU": '5',
    "DI": '6'
}

# Replace values in the Action_Label column
data['Action_Label'] = data['Action_Label'].replace(label_mapping)

# Filter the dataset for included classes
filtered_data = data[data['Action_Label'].isin(classes)]
#filtered_data = data
# Separate features and labels
X = filtered_data.drop(columns=['Action_Label', 'Image_Name']).values  # Features
y = filtered_data['Action_Label'].values  # Labels
classes = filtered_data['Action_Label'].unique()

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
X_selected = X_scaled  # If you don't want to use PCA

# Stratified KFold cross-validation
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

# Store the results
fold_accuracies = []
fold_classification_reports = []
fold_confusion_matrices = []

# Initialize an empty confusion matrix for averaging
avg_conf_matrix = np.zeros((len(classes), len(classes)))

for fold, (train_idx, val_idx) in enumerate(skf.split(X_selected, y_encoded), 1):
    print(f"\nTraining fold {fold}/{N_FOLDS}...")
    
    # Split data into training and validation sets
    X_train, X_val = X_selected[train_idx], X_selected[val_idx]
    y_train, y_val = y_one_hot[train_idx], y_one_hot[val_idx]
    
    # Build the neural network model
    model = Sequential([
        Dense(32, input_dim=X_train.shape[1], activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(16, activation='relu', kernel_regularizer=l2(0.001)),
        BatchNormalization(),
        Dropout(0.4),
        Dense(y_train.shape[1], activation='softmax')
    ])

    # Compile the model
    optimizer = Adam(learning_rate=0.0007)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Save the best model during training
    best_model_path = f"models/cv/best_model_fold{fold}.h5"
    checkpoint = ModelCheckpoint(best_model_path, save_best_only=True, monitor="val_accuracy", verbose=1)
    
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=500, batch_size=32, callbacks=[checkpoint], verbose=1)
    
    # Load the best model for this fold
    loaded_model = load_model(best_model_path)
    
    # Evaluate the model on the validation set
    loss, accuracy = loaded_model.evaluate(X_val, y_val)
    print(f"Fold {fold} - Validation Accuracy: {accuracy * 100:.2f}%")
    fold_accuracies.append(accuracy)
    
    # Predict and generate the classification report
    y_pred_one_hot = loaded_model.predict(X_val)
    y_pred = np.argmax(y_pred_one_hot, axis=1)
    y_true = np.argmax(y_val, axis=1)
    
    fold_classification_reports.append(classification_report(y_true, y_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    fold_confusion_matrices.append(conf_matrix)

    # Normalize the confusion matrix
    conf_matrix_normalized = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
    avg_conf_matrix += conf_matrix_normalized

    # Optionally plot confusion matrix for the current fold
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=label_encoder.classes_,
                yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Normalized Confusion Matrix - Fold {fold}')
    plt.show()

# Compute the average confusion matrix
avg_conf_matrix /= N_FOLDS

# Print the average confusion matrix
print("\nAverage Normalized Confusion Matrix Across All Folds:")
plt.figure(figsize=(10, 8))
sns.heatmap(avg_conf_matrix, annot=True, fmt='.2f', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Average Normalized Confusion Matrix Across All Folds')
plt.show()

# Print overall cross-validation results
print("\nOverall Cross-Validation Results:")
print(f"Mean Accuracy: {np.mean(fold_accuracies) * 100:.2f}%")
print(f"Standard Deviation of Accuracy: {np.std(fold_accuracies) * 100:.2f}%")

# Optionally save the results to a file
with open("models/cv/cross_validation_results.pkl", 'wb') as file:
    pickle.dump({
        'accuracies': fold_accuracies,
        'classification_reports': fold_classification_reports,
        'confusion_matrices': fold_confusion_matrices,
        'avg_confusion_matrix': avg_conf_matrix
    }, file)
