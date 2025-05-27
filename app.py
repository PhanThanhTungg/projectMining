from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
from customeLGBM import TurboLightGBMClassifier
import traceback
from flask_cors import CORS

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

model = None
scaler = None
disease_mapping = None
feature_names = None

def load_model_and_scaler():
    """Load model, scaler, disease mapping và feature names"""
    global model, scaler, disease_mapping, feature_names
    
    try:
        model = joblib.load('custom_lgbm_model.joblib')
        
        scaler = joblib.load('scaler.joblib')
        
        disease_mapping = joblib.load('disease_mapping.joblib')
        
        try:
            feature_names = joblib.load('feature_names.joblib')
        except:
            feature_names = [f"feature_{i+1}" for i in range(24)]
        
        print("✅ Model, scaler, disease mapping và feature names đã được load!")
        return True
    except Exception as e:
        print(f"Lỗi khi load model: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model hoặc scaler chưa được load. Vui lòng khởi động lại server.'
            }), 500
        
        data = request.get_json()
        
        if data is None:
            return jsonify({'error': 'Không có dữ liệu được gửi'}), 400
        
        if 'features' in data:
            features = np.array(data['features']).reshape(1, -1)
        elif 'data' in data:
            df = pd.DataFrame([data['data']])
            features = df.values
        else:
            return jsonify({
                'error': 'Format dữ liệu không đúng. Cần "features" array hoặc "data" object'
            }), 400
        
        features_scaled = scaler.transform(features)
        
        prediction = model.predict(features_scaled)[0]
        prediction_proba = model.predict_proba(features_scaled)[0]
        
        disease_name = disease_mapping.get(prediction, f"Unknown_{prediction}")
        
        confidence_scores = {}
        for class_idx, prob in enumerate(prediction_proba):
            disease_class = disease_mapping.get(class_idx, f"Unknown_{class_idx}")
            confidence_scores[disease_class] = float(prob)
        
        sorted_confidence = dict(sorted(confidence_scores.items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return jsonify({
            'predicted_disease': disease_name,
            'predicted_class': int(prediction),
            'confidence': float(prediction_proba[prediction]),
            'all_probabilities': sorted_confidence,
            'status': 'success'
        })
        
    except Exception as e:
        error_msg = f"Lỗi trong quá trình prediction: {str(e)}"
        print(f"{error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model hoặc scaler chưa được load.'
            }), 500
        
        data = request.get_json()
        
        if 'samples' not in data:
            return jsonify({
                'error': 'Cần trường "samples" chứa list các samples'
            }), 400
        
        samples = data['samples']
        features_list = []
        
        for sample in samples:
            if isinstance(sample, list):
                if len(sample) != 24:
                    return jsonify({
                        'error': f'Mỗi sample cần đúng 24 features, nhận được {len(sample)}'
                    }), 400
                features_list.append(sample)
            elif isinstance(sample, dict):
                feature_array = []
                for feature_name in feature_names:
                    if feature_name in sample:
                        feature_array.append(sample[feature_name])
                    else:
                        return jsonify({
                            'error': f'Thiếu feature: {feature_name}'
                        }), 400
                features_list.append(feature_array)
            else:
                return jsonify({
                    'error': 'Mỗi sample phải là list hoặc dictionary'
                }), 400
        
        features = np.array(features_list)
        
        features_scaled = scaler.transform(features)
        
        # Prediction
        predictions = model.predict(features_scaled)
        predictions_proba = model.predict_proba(features_scaled)
        
        results = []
        for i, (original_features, pred, proba) in enumerate(zip(features, predictions, predictions_proba)):
            result_object = {}
            
            for j, feature_name in enumerate(feature_names):
                result_object[feature_name] = float(original_features[j])
            
            result_object["target"] = disease_mapping.get(pred, f"Unknown_{pred}")
            result_object["confidence"] = float(proba[pred])
            
            results.append(result_object)
        
        return jsonify(results)
        
    except Exception as e:
        error_msg = f"Lỗi trong batch prediction: {str(e)}"
        print(f"{error_msg}")
        return jsonify({'error': error_msg}), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({'error': 'Model chưa được load'}), 500
    
    return jsonify({
        'model_type': 'TurboLightGBMClassifier',
        'n_classes': len(disease_mapping) if disease_mapping else 0,
        'disease_classes': disease_mapping,
        'n_estimators': model.n_estimators,
        'learning_rate': model.learning_rate,
        'max_depth': model.max_depth
    })

if __name__ == '__main__':
    print("Khởi động API server...")
    if load_model_and_scaler():
        print("API sẵn sàng!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Không thể khởi động API do lỗi load model")