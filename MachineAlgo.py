import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the updated dataset
file_path = "Updated_Dataset.csv"
df = pd.read_csv(file_path)

# Drop unnecessary columns (like index if present)
df = df.drop(columns=['Unnamed: 0'], errors='ignore')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['HTTP Method', 'API Endpoint', 'User-Agent', 'Classification Label', 
                    'Generalized API Endpoint', 'Time of Day']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store the encoders for future use

# Save label encoders
label_encoder_file_path = "label_encoder.pkl"
joblib.dump(label_encoders, label_encoder_file_path)

# Select features and target
features = [
    'Rate Limiting', 'Endpoint Entropy', 'HTTP Method', 'API Endpoint',
    'HTTP Status', 'User-Agent', 'Token Used', 'Generalized API Endpoint', 'Method_POST', 'Time of Day'
]

X = df[features]
y = df['Classification Label']  # Target: Legitimate vs. Potential Misuse

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler
scaler_file_path = "scaler.pkl"
joblib.dump(scaler, scaler_file_path)

# Train an XGBoost classifier with regularization and adjusted parameters to prevent overfitting
model = xgb.XGBClassifier(
    n_estimators=50,         # Reduce trees to prevent overfitting
    max_depth=2,             # Reduce model complexity
    learning_rate=0.1,       # Moderate learning rate for generalization
    subsample=0.7,           # Use only 70% of data per tree
    colsample_bytree=0.7,    # Use only 70% of features per tree
    alpha=5,                 # Reduce L1 regularization
    reg_lambda=5,            # Reduce L2 regularization
    objective='binary:logistic',
    use_label_encoder=False,
    eval_metric='logloss'
)

# Train model without early stopping
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Get feature importance for explanations
feature_importances = model.feature_importances_
important_factors = {features[i]: feature_importances[i] for i in range(len(features))}

# Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Save the trained model as a pkl file
model_file_path = "XGBoost_Anomaly_Model.pkl"
with open(model_file_path, 'wb') as model_file:
    pickle.dump(model, model_file)

# Output model performance and feature importance
print("Model Accuracy:", accuracy)
print("Classification Report:\n", report)
print("Important Factors Contributing to Classification:")
for key, value in sorted(important_factors.items(), key=lambda x: x[1], reverse=True):
    print(f"{key}: {value:.4f}")
