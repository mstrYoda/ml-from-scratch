from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import pickle
import json
from flask import Flask, request, jsonify
import numpy as np

# Load and prepare data
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# Create and train pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

pipeline.fit(X_train, y_train)

# 1. Save model using joblib
joblib.dump(pipeline, 'model_joblib.pkl')

# 2. Save model using pickle
with open('model_pickle.pkl', 'wb') as f:
    pickle.dump(pipeline, f)

# 3. Save model metadata
model_metadata = {
    'feature_names': iris.feature_names,
    'target_names': iris.target_names,
    'model_params': pipeline.get_params()
}

with open('model_metadata.json', 'w') as f:
    json.dump(model_metadata, f)

# 4. Create Flask API
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = np.array(data['features']).reshape(1, -1)
        prediction = pipeline.predict(features)
        probability = pipeline.predict_proba(features)
        
        response = {
            'prediction': int(prediction[0]),
            'class_name': iris.target_names[prediction[0]],
            'probability': probability[0].tolist()
        }
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

# 5. Model loading and validation
def load_and_validate_model():
    # Load model using joblib
    loaded_model = joblib.load('model_joblib.pkl')
    
    # Validate model
    test_score = loaded_model.score(X_test, y_test)
    print(f"Loaded model test score: {test_score:.4f}")
    
    return loaded_model

if __name__ == '__main__':
    # Test model loading
    loaded_model = load_and_validate_model()
    
    # Start Flask app
    app.run(debug=True) 