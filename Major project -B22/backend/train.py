import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
df = pd.read_csv("large_flight_data.csv")
df.columns = df.columns.str.strip().str.lower()

# Feature Engineering
df["vertical_speed_change_rate"] = df["vertical_speed"].diff().fillna(0)
df["rate_of_descent_change_rate"] = df["rate_of_descent"].diff().fillna(0)
df["pitch_variability_rate"] = df["pitch_variability"].diff().fillna(0)
df["roll_angle_variability_rate"] = df["roll_angle_variability"].diff().fillna(0)
df["altitude_change_rate"] = df["altitude"].diff().fillna(0)

# Split features and target
X = df.drop(columns=['hard_landing'])
y = df['hard_landing']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Resample the training data to balance the classes
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Train the model
fnn_model = MLPClassifier(hidden_layer_sizes=(32, 16), activation='relu', solver='adam', max_iter=500, random_state=42)
fnn_model.fit(X_train_scaled, y_train_resampled)

# Evaluate the model
y_pred = fnn_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Print Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the trained model and scaler
joblib.dump(fnn_model, 'fnn_hard_landing_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Soft Landing', 'Hard Landing'], yticklabels=['Soft Landing', 'Hard Landing'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, fnn_model.predict_proba(X_test_scaled)[:, 1])
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='b', label=f'ROC Curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')
plt.show()

# Print Model and Scaler Saved Message
print("Model and scaler have been saved successfully!")