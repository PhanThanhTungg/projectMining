from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import warnings
from customeLGBM import TurboLightGBMClassifier, evaluate_model
import traceback
from flask_cors import CORS
from pymongo import MongoClient
from datetime import datetime
from sklearn.pipeline import Pipeline
import uuid

import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import pandas as pd

warnings.filterwarnings("ignore")

app = Flask(__name__)
CORS(app)

model = None
model_lightgbm = None
scaler = None
disease_mapping = None
feature_names = None
collection = None

def load_model_and_scaler():
    """Load model, scaler, disease mapping và feature names"""
    global model, model_lightgbm, scaler, disease_mapping, feature_names, collection

    try:
        model = joblib.load('./Models/custom_lgbm_model.joblib')

        model_lightgbm = joblib.load('./Models/lightgbm_model.joblib')
        
        scaler = joblib.load('./Models/scaler.joblib')
        
        disease_mapping = joblib.load('./Models/disease_mapping.joblib')
        
        try:
            feature_names = joblib.load('./Models/feature_names.joblib')
        except:
            feature_names = [f"feature_{i+1}" for i in range(24)]
        
        print("✅ Model, scaler, disease mapping và feature names đã được load!")
        
        client = MongoClient("mongodb+srv://TungConnectDTB:TungConnectDTB@cluster0.berquuj.mongodb.net/")
        db = client['mining']
        collection = db['samples']
        
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

        # LightGBM prediction
        lightgbm_prediction = model_lightgbm.predict(features_scaled)[0]
        lightgbm_proba = model_lightgbm.predict_proba(features_scaled)[0]
        lightgbm_disease_name = disease_mapping.get(lightgbm_prediction, f"Unknown_{lightgbm_prediction}")
        
        sorted_lightgbm_confidence = dict(sorted(
            {disease_mapping.get(i, f"Unknown_{i}"): float(prob) for i, prob in enumerate(lightgbm_proba)}.items(),
            key=lambda x: x[1], reverse=True
        ))
        
        return jsonify({
            'custom':{
                'predicted_disease': disease_name,
                'predicted_class': int(prediction),
                'confidence': float(prediction_proba[prediction]),
                'all_probabilities': sorted_confidence,
                'status': 'success'
            },
            'lightgbm':{
                'predicted_disease': lightgbm_disease_name,
                'predicted_class': int(lightgbm_prediction),
                'confidence': float(lightgbm_proba[lightgbm_prediction]),
                'all_probabilities': sorted_lightgbm_confidence,
                'status': 'success'
            }
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
        
        if 'file' not in request.files:
            return jsonify({
                'error': 'Cần upload file CSV'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Không có file nào được chọn'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'error': 'File phải có định dạng CSV'
            }), 400
        
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'error': f'Lỗi khi đọc file CSV: {str(e)}'
            }), 400
        
        if df.empty:
            return jsonify({
                'error': 'File CSV rỗng'
            }), 400
        
        missing_features = []
        for feature_name in feature_names:
            if feature_name not in df.columns:
                missing_features.append(feature_name)
        
        if missing_features:
            return jsonify({
                'error': f'Thiếu các feature sau trong file CSV: {missing_features}'
            }), 400
        
        features_df = df[feature_names]
        
        if features_df.isnull().any().any():
            return jsonify({
                'error': 'File CSV chứa giá trị null/NaN'
            }), 400
        
        features = features_df.values
        
        features_scaled = scaler.transform(features)
        
        predictions = model.predict(features_scaled)
        predictions_proba = model.predict_proba(features_scaled)

        #lightgbm predictions
        lightgbm_predictions = model_lightgbm.predict(features_scaled)
        lightgbm_predictions_proba = model_lightgbm.predict_proba(features_scaled)
        
        results = []
        for i, (original_features, pred, proba) in enumerate(zip(features, predictions, predictions_proba)):
            result_object = {}
            result_object['custom'] = {}
            
            for j, feature_name in enumerate(feature_names):
                result_object['custom'][feature_name] = float(original_features[j])
            
            result_object["custom"]["predicted_disease"] = disease_mapping.get(pred, f"Unknown_{pred}")
            result_object["custom"]["predicted_class"] = int(pred)
            result_object["custom"]["confidence"] = float(proba[pred])
            
            confidence_scores = {}
            for class_idx, prob in enumerate(proba):
                disease_class = disease_mapping.get(class_idx, f"Unknown_{class_idx}")
                confidence_scores[disease_class] = float(prob)
            
            result_object["custom"]["all_probabilities"] = dict(sorted(confidence_scores.items(), 
                                                           key=lambda x: x[1], reverse=True))
            
            # LightGBM 
            result_object['lightgbm'] = {}
            result_object["lightgbm"]["predicted_disease"] = disease_mapping.get(lightgbm_predictions[i], f"Unknown_{lightgbm_predictions[i]}")
            result_object["lightgbm"]["predicted_class"] = int(lightgbm_predictions[i])
            result_object["lightgbm"]["confidence"] = float(lightgbm_predictions_proba[i][lightgbm_predictions[i]])
            lightgbm_confidence_scores = {}
            for class_idx, prob in enumerate(lightgbm_predictions_proba[i]):
                disease_class = disease_mapping.get(class_idx, f"Unknown_{class_idx}")
                lightgbm_confidence_scores[disease_class] = float(prob)
            result_object["lightgbm"]["all_probabilities"] = dict(sorted(lightgbm_confidence_scores.items(),
                                                           key=lambda x: x[1], reverse=True))
            
            results.append(result_object)
        
        return jsonify({
            'status': 'success',
            'total_samples': len(results),
            'results': results
        })
        
    except Exception as e:
        error_msg = f"Lỗi trong batch prediction: {str(e)}"
        print(f"{error_msg}")
        print(traceback.format_exc())
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

@app.route('/upload_csv_data', methods=['POST'])
def upload_csv_data():
    """Upload file CSV và lưu data vào MongoDB"""
    try:
        if collection is None:
            return jsonify({'error': 'Chưa kết nối MongoDB'}), 500
        
        # Kiểm tra file upload
        if 'file' not in request.files:
            return jsonify({
                'error': 'Cần upload file CSV'
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'error': 'Không có file nào được chọn'
            }), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({
                'error': 'File phải có định dạng CSV'
            }), 400
        
        # Đọc file CSV
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'error': f'Lỗi khi đọc file CSV: {str(e)}'
            }), 400
        
        if df.empty:
            return jsonify({
                'error': 'File CSV rỗng'
            }), 400
        
        print(f" File CSV có {len(df)} dòng và {len(df.columns)} cột")
        print(f"Columns: {list(df.columns)}")
        
        # Kiểm tra target column
        target_column = None
        possible_targets = ['target', 'Target', 'disease', 'Disease', 'label', 'Label']
        
        for col in possible_targets:
            if col in df.columns:
                target_column = col
                break
        
        if target_column is None:
            return jsonify({
                'error': f'Không tìm thấy cột target. Cần một trong các cột: {possible_targets}'
            }), 400
        
        # Kiểm tra 24 features
        feature_columns = [col for col in df.columns if col != target_column]
        
        if len(feature_columns) != 24:
            return jsonify({
                'error': f'Cần đúng 24 features, tìm thấy {len(feature_columns)} features. Target column: {target_column}'
            }), 400
        
        # Validate data types và missing values
        feature_df = df[feature_columns]
        target_series = df[target_column]
        
        # Kiểm tra missing values
        if feature_df.isnull().any().any():
            null_counts = feature_df.isnull().sum()
            null_features = null_counts[null_counts > 0].to_dict()
            return jsonify({
                'error': f'File CSV chứa giá trị null/NaN trong features: {null_features}'
            }), 400
        
        if target_series.isnull().any():
            return jsonify({
                'error': f'Cột target "{target_column}" chứa giá trị null/NaN'
            }), 400
        
        # Kiểm tra data types cho features (phải là số)
        non_numeric_features = []
        for col in feature_columns:
            if not pd.api.types.is_numeric_dtype(feature_df[col]):
                try:
                    pd.to_numeric(feature_df[col])
                except:
                    non_numeric_features.append(col)
        
        if non_numeric_features:
            return jsonify({
                'error': f'Các features sau không phải là số: {non_numeric_features}'
            }), 400
        
        # Chuẩn bị data để insert vào MongoDB
        records_to_insert = []
        insert_errors = []
        
        for index, row in df.iterrows():
            try:
                record = {}
                
                # Map features theo thứ tự feature_names
                for i, feature_name in enumerate(feature_names):
                    if i < len(feature_columns):
                        feature_value = float(row[feature_columns[i]])
                        record[feature_name] = feature_value
                    else:
                        # Nếu thiếu features, báo lỗi
                        raise ValueError(f"Thiếu feature thứ {i+1}")
                
                # Add target
                record['Disease'] = str(row[target_column])

                # duplicate_check = collection.find_one({
                #     'Disease': record['Disease'],
                #     **{feature_name: record[feature_name] for feature_name in feature_names}
                # })
                # if duplicate_check:
                #     print(f"Dòng {index + 1} đã tồn tại trong MongoDB, bỏ qua.")
                #     continue
                
                records_to_insert.append(record)
                
            except Exception as e:
                insert_errors.append(f"Dòng {index + 1}: {str(e)}")
        
        if insert_errors:
            return jsonify({
                'error': 'Lỗi khi xử lý dữ liệu',
                'details': insert_errors[:10]  # Chỉ hiển thị 10 lỗi đầu
            }), 400
        
        # Insert vào MongoDB
        if records_to_insert:
            try:
                result = collection.insert_many(records_to_insert)
                inserted_count = len(result.inserted_ids)
                
                return jsonify({
                    'status': 'success'
                }), 201
                
            except Exception as e:
                return jsonify({
                    'error': f'Lỗi khi insert vào MongoDB: {str(e)}'
                }), 500
        else:
            return jsonify({
                'error': 'Không có data hợp lệ để insert'
            }), 400
            
    except Exception as e:
        error_msg = f"Lỗi khi upload CSV: {str(e)}"
        print(f"{error_msg}")
        print(traceback.format_exc())
        return jsonify({'error': error_msg}), 500

@app.route("/train_model", methods=['POST'])
def train_model():
    try:
        cursor = collection.find({})
        data_list = list(cursor)
        if not data_list:
            raise ValueError("Không có dữ liệu trong MongoDB")
        print(f"Đã tải {len(data_list)} records từ MongoDB")
        data = pd.DataFrame(data_list)
        data.drop('_id', axis=1, inplace=True, errors='ignore')
        print(f"DataFrame shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
        if 'Disease' not in data.columns:
            raise ValueError("Không tìm thấy cột 'target' trong dữ liệu")
        
        data.Disease = data.Disease.astype('category')
        disease_mapping = dict(enumerate(data['Disease'].cat.categories))
        print(f"Disease classes: {disease_mapping}")
        
        feature_names = data.drop('Disease', axis=1).columns.tolist()
        print(f"Feature names: {feature_names}")
        print(f"Number of features: {len(feature_names)}")
        
        data.Disease = data.Disease.cat.codes.values
        X = data.drop('Disease', axis=1).values
        y = data['Disease'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        turbo_lgbm = TurboLightGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=0.1,
            max_bins=255,
            n_jobs=-1, 
            random_state=42
        )
        
        start_time = time.time()
        turbo_lgbm.fit(X_train_scaled, y_train)

        # Lưu model, scaler, disease mapping và feature names
        joblib.dump(turbo_lgbm, './Models/custom_lgbm_model.joblib')
        joblib.dump(scaler, './Models/scaler.joblib')
        joblib.dump(disease_mapping, './Models/disease_mapping.joblib')
        joblib.dump(feature_names, './Models/feature_names.joblib')

        print("Đã lưu model, scaler, disease mapping và feature names!")
        
        turbo_time = time.time() - start_time
        
        turbo_results = evaluate_model(turbo_lgbm, X_test_scaled, y_test, "TURBO LightGBM")

        # Train LightGBM model
        best_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LGBMClassifier(
                learning_rate=0.1,
                max_depth=-1,
                min_child_samples=50,
                n_estimators=300,
                num_leaves=50,
                random_state=42,
                verbosity=-1
            ))
        ])
        best_pipeline.fit(X_train_scaled, y_train)
        joblib.dump(best_pipeline, './Models/lightgbm_model.joblib')

        return jsonify({
            "custom_lgbm":{
                'status': 'success',
                'message': 'Đã train model thành công',
                'training_time': turbo_time,
                'evaluation_results': turbo_results
            },
            "lightgbm":{
                'status': 'success',
                'message': 'Đã train LightGBM model thành công',
                'training_time': time.time() - start_time,
                'evaluation_results': evaluate_model(best_pipeline, X_test_scaled, y_test, "LightGBM")
            }
        }), 200
    except Exception as e:
        print(f"Lỗi : {str(e)}")
        return jsonify({'error': f'Lỗi: {str(e)}'}), 500

if __name__ == '__main__':
    print("Khởi động API server...")
    if load_model_and_scaler():
        print("API sẵn sàng!")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("Không thể khởi động API do lỗi load model")