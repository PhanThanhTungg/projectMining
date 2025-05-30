import pandas as pd
import warnings
from customeLGBM import TurboLightGBMClassifier, evaluate_model
import time
import joblib
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    import pandas as pd
    
    print("Loading dataset...")
    data = pd.read_csv('./Data/iniDataset.csv')
    
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
    joblib.dump(turbo_lgbm, 'custom_lgbm_model.joblib')
    joblib.dump(scaler, 'scaler.joblib')
    joblib.dump(disease_mapping, 'disease_mapping.joblib')
    joblib.dump(feature_names, 'feature_names.joblib')
    
    print("Đã lưu model, scaler, disease mapping và feature names!")
    
    turbo_time = time.time() - start_time
    
    turbo_results = evaluate_model(turbo_lgbm, X_test_scaled, y_test, "TURBO LightGBM")