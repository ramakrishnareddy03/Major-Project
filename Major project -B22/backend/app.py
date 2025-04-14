from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import io
import base64

# Load the trained model and scaler
fnn_model = joblib.load("fnn_hard_landing_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load the dataset for metrics and evaluation
df = pd.read_csv("large_flight_data.csv")
df.columns = df.columns.str.strip().str.lower()

# Feature Engineering for evaluation
df["vertical_speed_change_rate"] = df["vertical_speed"].diff().fillna(0)
df["rate_of_descent_change_rate"] = df["rate_of_descent"].diff().fillna(0)
df["pitch_variability_rate"] = df["pitch_variability"].diff().fillna(0)
df["roll_angle_variability_rate"] = df["roll_angle_variability"].diff().fillna(0)
df["altitude_change_rate"] = df["altitude"].diff().fillna(0)

# Split features and target for evaluation
X = df.drop(columns=['hard_landing'])
y = df['hard_landing']

# Function to preprocess input data (same as training)
def preprocess_input(data):
    df_input = pd.DataFrame([data])
    df_input["vertical_speed_change_rate"] = df_input["vertical_speed"].diff().fillna(0)
    df_input["rate_of_descent_change_rate"] = df_input["rate_of_descent"].diff().fillna(0)
    df_input["pitch_variability_rate"] = df_input["pitch_variability"].diff().fillna(0)
    df_input["roll_angle_variability_rate"] = df_input["roll_angle_variability"].diff().fillna(0)
    df_input["altitude_change_rate"] = df_input["altitude"].diff().fillna(0)
    df_input = df_input.fillna(0)
    features = df_input.drop(columns=['hard_landing'], errors='ignore')
    scaled_features = scaler.transform(features)
    return scaled_features

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the request
        data = request.get_json()
        data = {key: float(value) for key, value in data.items()}  # Ensure floats

        # Debug: Print the received data
        print(f"Received Data: {data}")

        # Preprocess input
        processed_data = preprocess_input(data)

        # Make prediction
        prediction = fnn_model.predict(processed_data)

        # Debug: Print the prediction
        print(f"Prediction: {prediction}")

        # Prepare result
        result = {'prediction': 'Hard Landing' if prediction[0] == 1 else 'Soft Landing'}
        return jsonify(result)

    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging the error
        return jsonify({'error': str(e)})

# Route for Evaluation Metrics
@app.route('/metrics', methods=['GET'])
def metrics():
    try:
        # Resample and scale the dataset for evaluation
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Evaluate the model
        X_test_scaled = scaler.transform(X_test)
        y_pred = fnn_model.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)  # Convert to dictionary

        precision = report["1"]["precision"]  # Hard landing class (label 1)
        recall = report["1"]["recall"]
        f1_score = report["1"]["f1-score"]

        # Create confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        cm_image = plot_confusion_matrix(cm)

        # Create ROC curve plot
        roc_image = plot_roc_curve(y_test, X_test_scaled)

        return jsonify({
            'accuracy': f'{accuracy * 100:.2f}%',
            'precision': f'{precision:.2f}',
            'recall': f'{recall:.2f}',
            'f1_score': f'{f1_score:.2f}',
            'confusion_matrix': cm_image,
            'roc_curve': roc_image
        })

    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({'error': str(e)})


# Function to plot confusion matrix
def plot_confusion_matrix(cm):
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Soft Landing', 'Hard Landing'], yticklabels=['Soft Landing', 'Hard Landing'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    # Save the plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_b64

# Function to plot ROC curve
def plot_roc_curve(y_test, X_test_scaled):
    fpr, tpr, thresholds = roc_curve(y_test, fnn_model.predict_proba(X_test_scaled)[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='b', label=f'ROC Curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_title("Receiver Operating Characteristic (ROC) Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc='lower right')

    # Save the plot to a BytesIO buffer and encode as base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    return img_b64

if __name__ == '__main__':
    app.run(debug=True)