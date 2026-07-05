import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir) if os.path.basename(base_dir) == "src" else base_dir
        
    input_file = os.path.join(project_root, "data", "processed", "zamora_houses_clean.csv")
    model_dir = os.path.join(project_root, "outputs", "models")
    os.makedirs(model_dir, exist_ok=True)

    print("🤖 Starting Model Training Pipeline...")
    df = pd.read_csv(input_file)

    # 1. Base Cleaning & Boundary Filtering
    df_ml = df[(df['sq_meters'] > 0) & (df['bedrooms'] > 0)].copy()
    
    if 'latitude' in df_ml.columns and 'longitude' in df_ml.columns:
        df_ml = df_ml[
            (df_ml['latitude'].between(41.1, 42.2)) & 
            (df_ml['longitude'].between(-6.3, -5.3))
        ]
    else:
        print("❌ Error: Coordinates missing.")
        return
    
    # Tighten target outliers slightly to stabilize variance on small sample size
    price_cap = df_ml['price'].quantile(0.97)
    df_ml = df_ml[df_ml['price'] < price_cap].copy()
    
    # 2. Advanced Feature Generation
    df_ml['m2_per_bedroom'] = df_ml['sq_meters'] / df_ml['bedrooms']
    df_ml['size_per_room'] = df_ml['sq_meters'] / (df_ml['bedrooms'] + 1)
    if 'bathrooms' in df_ml.columns:
        df_ml['bath_to_bed_ratio'] = df_ml['bathrooms'] / (df_ml['bedrooms'] + 1)
        
    # Spatial center proximity metric
    zamora_lat, zamora_lon = 41.5063, -5.7446
    df_ml['distance_to_center'] = np.sqrt(
        (df_ml['latitude'] - zamora_lat)**2 + 
        (df_ml['longitude'] - zamora_lon)**2
    )
    df_ml['size_distance_interaction'] = df_ml['sq_meters'] * df_ml['distance_to_center']
    
    # One-hot encoding structural types
    cat_cols = [c for c in ['property_type', 'type'] if c in df_ml.columns]
    if cat_cols:
        df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)

    if 'location' in df_ml.columns:
        df_ml = df_ml.drop(columns=['location'])
        
    df_ml = df_ml.select_dtypes(include=[np.number, bool])
    
    # 3. Train/Test Split
    X = df_ml.drop(columns=['price'])
    y = df_ml['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set size: {X_train.shape[0]} samples | Test set size: {X_test.shape[0]} samples\n")

    # 📈 NEW ADDITION: Scale features using RobustScaler (handles real estate outliers cleanly)
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Convert targets to log spaces
    y_train_log = np.log1p(y_train)

    # 4. Train Random Forest
    print("🌲 Training Random Forest Regressor...")
    rf_model = RandomForestRegressor(
        n_estimators=250, 
        max_depth=10, 
        min_samples_split=3, 
        random_state=42
    )
    rf_model.fit(X_train_scaled, y_train_log)
    rf_preds = np.expm1(rf_model.predict(X_test_scaled))

    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)

    # 5. Train XGBoost (Optimized tuning parameters)
    print("🚀 Training XGBoost Regressor...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=400, 
        learning_rate=0.02, 
        max_depth=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=1.2,
        reg_lambda=2.0,
        random_state=42
    )
    xgb_model.fit(X_train_scaled, y_train_log)
    xgb_preds = np.expm1(xgb_model.predict(X_test_scaled))

    xgb_mae = mean_absolute_error(y_test, xgb_preds)
    xgb_r2 = r2_score(y_test, xgb_preds)

    # 6. Compare Performance
    print("\n==================================================")
    print("             MODEL COMPETITION RESULTS            ")
    print("==================================================")
    print(f"Random Forest -> MAE: {rf_mae:,.2f} € | R² Score: {rf_r2:.4f}")
    print(f"XGBoost       -> MAE: {xgb_mae:,.2f} € | R² Score: {xgb_r2:.4f}")
    print("==================================================")

    # 7. Save Assets
    plot_dir = os.path.join(project_root, "outputs", "plots")
    os.makedirs(plot_dir, exist_ok=True)

    def save_eval_plot(y_true, y_pred, name):
        plt.figure(figsize=(7, 7))
        sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, color='#2c3e50')
        ideal = [y_true.min(), y_true.max()]
        plt.plot(ideal, ideal, color='#e74c3c', linestyle='--', label='Perfect Prediction')
        plt.title(f"Model Accuracy Report Card ({name})")
        plt.xlabel("Actual Price (€)")
        plt.ylabel("Predicted Price (€)")
        plt.gca().xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}€"))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}€"))
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "model_predictions_vs_actual.png"))
        plt.close()

    if rf_r2 > xgb_r2:
        print(f"🏆 Random Forest wins! Saving model...")
        joblib.dump(rf_model, os.path.join(model_dir, "best_house_predictor.pkl"))
        save_eval_plot(y_test, rf_preds, "Random Forest")
    else:
        print(f"🏆 XGBoost wins! Saving model...")
        joblib.dump(xgb_model, os.path.join(model_dir, "best_house_predictor.pkl"))
        save_eval_plot(y_test, xgb_preds, "XGBoost")
        
    # Save scaler transformation parameters so future inference inputs match
    joblib.dump(scaler, os.path.join(model_dir, "robust_scaler.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "model_features.pkl"))
    print(f"✅ Success! Winning model assets saved to: {model_dir}")

if __name__ == "__main__":
    main()