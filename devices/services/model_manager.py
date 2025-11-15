import os
import pandas as pd
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from time_series_predictor import LSTMModel, TimeSeriesPredictor

class ModelManager:
    def __init__(self, model_directory="trained_models/"):
        self.model_directory = model_directory
        os.makedirs(self.model_directory, exist_ok=True)
        self.current_models = {}  # {model_name: latest_version}

    # ذخیره مدل و scaler با نسخه‌بندی
    def save_pytorch_lstm(self, model, scaler, model_name: str):
        version = torch.tensor([torch.randint(0,99999,(1,))]).item()
        model_path = os.path.join(self.model_directory, f"{model_name}_v{version}.pth")
        scaler_path = os.path.join(self.model_directory, f"{model_name}_v{version}_scaler.pkl")
        meta_path = os.path.join(self.model_directory, f"{model_name}_v{version}_meta.json")
        
        torch.save(model.state_dict(), model_path)
        torch.save(scaler, scaler_path)
        metadata = {
            "model_name": model_name,
            "version": version,
            "saved_at": str(torch.tensor([torch.randint(0,99999,(1,))]).item())
        }
        with open(meta_path, "w") as f:
            import json
            json.dump(metadata, f, indent=4)

        self.current_models[model_name] = (version, model_path, scaler_path)
        print(f"Model '{model_name}' saved with version {version}")


    # لود مدل و scaler
    def load_pytorch_lstm(self, model_class, model_name: str, input_size: int):
        if model_name not in self.current_models:
            raise ValueError(f"No saved version for {model_name}")
        version, model_path, scaler_path = self.current_models[model_name]
        model = model_class(input_size=input_size)
        model.load_state_dict(torch.load(model_path))
        scaler = torch.load(scaler_path)
        return model, scaler, version

    # ارزیابی مدل
    def evaluate_lstm(self, model, X_test, y_test):
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32)
            preds = model(X_tensor).numpy().flatten()
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        print(f"Model performance: RMSE={rmse:.4f}, MAE={mae:.4f}")
        return rmse, mae

    # آموزش مجدد خودکار
    def retrain_lstm(self, model, scaler, model_name, X_train, y_train, baseline_rmse):
        print(f" Checking if model '{model_name}' needs retraining...")
        rmse, _ = self.evaluate_lstm(model, X_train, y_train)
        if rmse > baseline_rmse:
            print(f"RMSE has increased, model will be retrained...")
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                output = model(torch.tensor(X_train, dtype=torch.float32))
                loss = criterion(output, torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
                loss.backward()
                optimizer.step()
            print(" Retraining completed and new version will be saved")
            self.save_pytorch_lstm(model, scaler, model_name)
        else:
           print("Model does not require retraining")


# ------------------ مثال استفاده ------------------


df = pd.read_csv("ml_sensor_data_2000.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
series = df['temperature'].asfreq('h')  # سری زمانی دما با فرکانس ساعتی

predictor = TimeSeriesPredictor()
# scaler و features آماده هستن، sequence_length هم مشخص
sequence_length = 30

# آماده‌سازی داده‌ها
features_df = predictor.build_features(series)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_df.values)
predictor.scaler = scaler

X, y_values = [], []
for i in range(len(features_scaled) - sequence_length):
    X.append(features_scaled[i:i+sequence_length])
    y_values.append(features_scaled[i+sequence_length][0])
X = np.array(X)
y_values = np.array(y_values)

# آموزش اولیه LSTM
predictor.train_lstm_model(X, y_values, epochs=100, lr=0.0005)

# مدیریت مدل
manager = ModelManager()
manager.save_pytorch_lstm(predictor.lstm_model, predictor.scaler, "temperature_lstm")

# مانیتورینگ و retraining
baseline_rmse = 0.05  # فرضیه یا مقدار قبلی
manager.retrain_lstm(predictor.lstm_model, predictor.scaler, "temperature_lstm", X, y_values, baseline_rmse)
