import pandas as pd
import warnings
from customeLGBM import TurboLightGBMClassifier, evaluate_model
import time
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from lightgbm import LGBMClassifier
    import pandas as pd
    
    print("Loading dataset...")
    data = pd.read_csv('./Data/iniDataset.csv')
    
    # Prepare data
    data.Disease = data.Disease.astype('category')
    disease_mapping = dict(enumerate(data['Disease'].cat.categories))
    print(f"Disease classes: {disease_mapping}")
    
    data.Disease = data.Disease.cat.codes.values
    X = data.drop('Disease', axis=1).values
    y = data['Disease'].values
    
    # Train-test split
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
    turbo_time = time.time() - start_time
    
    turbo_results = evaluate_model(turbo_lgbm, X_test_scaled, y_test, "TURBO LightGBM")
    
    
    