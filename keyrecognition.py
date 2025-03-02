import pandas as pd
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import joblib

MODEL_DIR = Path(__file__).parent

csv_path = {
    'api_keys': str(MODEL_DIR / 'api_keys_dataset.csv')
}
# Load the dataset
data = pd.read_csv(csv_path['api_keys'])

# Feature Extraction
def extract_features(df):
    # Feature: Length of the key
    df['length'] = df['key'].apply(len)
    
    
    df['starts_with_sk_proj'] = df['key'].str.startswith('sk-proj-').astype(int)
    df['starts_with_ant'] = df['key'].str.startswith('sk-ant-api03-').astype(int)
    
    
    df['digit_ratio'] = df['key'].apply(lambda x: sum(c.isdigit() for c in x) / len(x))
    
    
    df['uppercase_ratio'] = df['key'].apply(lambda x: sum(c.isupper() for c in x) / len(x))
    
   
    df['non_alphanumeric_count'] = df['key'].apply(lambda x: sum(not c.isalnum() and c not in "-_" for c in x))
    
    # Feature:match for OpenAI format
    df['matches_openai_format'] = df['key'].apply(
        lambda x: bool(re.match(r'^sk-proj-([A-Za-z0-9]{10,20}-){4}[A-Za-z0-9]{10,20}$', x))
    ).astype(int)
    
    # Feature: match for Anthropic format
    df['matches_anthropic_format'] = df['key'].apply(
        lambda x: bool(re.match(r'^sk-ant-api03-[A-Za-z0-9_-]{95}$', x))
    ).astype(int)
    
    # Feature:match for Cohere format 
    df['matches_cohere_format'] = df['key'].apply(
        lambda x: bool(re.match(r'^[A-Za-z0-9]{40}$', x))
    ).astype(int)

    df['matches_poe_format'] = df['key'].apply(
        lambda x: bool(re.match(r'^[A-Za-z0-9_]{43}$', x))
    ).astype(int)
    
    return df

# Apply feature extraction
data = extract_features(data)

# Features and Labels
X = data[['length', 'starts_with_sk_proj', 'starts_with_ant',
          'digit_ratio', 'uppercase_ratio', 'non_alphanumeric_count',
          'matches_openai_format', 'matches_anthropic_format', 'matches_cohere_format','matches_poe_format']]
y = data['label']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))


# Save the model (optional, for deployment)
joblib.dump(clf, 'random_forest_api_key_model_v1.5.0.pkl')

# Function to test individual API keys
def test_api_key(key, model):
    # Extract features for the given key
    features = {
        'length': len(key),
        'starts_with_sk_proj': int(key.startswith('sk-proj-')),
        'starts_with_ant': int(key.startswith('sk-ant-api03-')),
        'digit_ratio': sum(c.isdigit() for c in key) / len(key),
        'uppercase_ratio': sum(c.isupper() for c in key) / len(key),
        'non_alphanumeric_count': sum(not c.isalnum() and c not in "-_" for c in key),
        'matches_openai_format': int(bool(re.match(r'^sk-proj-([A-Za-z0-9]{10,20}-){4}[A-Za-z0-9]{10,20}$', key))),
        'matches_anthropic_format': int(bool(re.match(r'^sk-ant-api03-[A-Za-z0-9_-]{95}$', key))),
        'matches_cohere_format': int(bool(re.match(r'^[A-Za-z0-9]{40}$', key))),
        'matches_poe_format': int(bool(re.match(r'^[A-Za-z0-9_]{43}$', key)))
    }
    
    # Convert to DataFrame for prediction
    input_features = pd.DataFrame([features])
    
    # Predict the label
    prediction = model.predict(input_features)[0]
    return prediction

# Example of testing individual keys
example_keys = [
    # Insert Valid OpenAI key here,
    # Insert an Invalid Cohere key example,
    # Insert Valid Anthropic key here,
    # Insert Valid Cohere key here,
    # Insert an Invalid Anthropic key example,
    # Insert a valid poe key here,
    # insert a invalid poe key
]

print("\nTesting Individual Keys:")
for key in example_keys:
    print(f"Key: {key}\nPrediction: {test_api_key(key, clf)}\n")
