import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ---------------- Dataset ----------------
class TimeSeriesDataset(Dataset):
    """کلاس دیتاست برای PyTorch که X و y را نگه می‌دارد"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)  # تبدیل X به Tensor
        self.y = torch.tensor(y, dtype=torch.float32)  # تبدیل y به Tensor

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]  # بازگرداندن یک نمونه

# ---------------- LSTM Model ----------------
class LSTMModel(nn.Module):
    """مدل LSTM ساده برای پیش‌بینی سری زمانی"""
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super(LSTMModel, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        # تبدیل خروجی lstm به یک عدد پیشبینی
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)           
        out = out[:, -1, :]              # آخرین زمان را می‌گیریم
        out = self.fc(out)           
        return out

# ---------------- Predictor ----------------
class TimeSeriesPredictor:
    """کلاس اصلی برای پیش‌بینی با ARIMA، Prophet و LSTM"""
    
#    انتخاب GPU یا CPU برای آموزش سریع
    def __init__(self, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')  # استفاده از GPU در صورت امکان
        self.scaler = None        # نگه داشتن scaler
        self.lstm_model = None    # نگه داشتن مدل LSTM

    def build_features(self, series: pd.Series):
        """ایجاد فیچرهای اضافی برای مدل LSTM"""
        df = pd.DataFrame({'y': series})
        df['ma_3'] = df['y'].rolling(3, min_periods=1).mean()      # میانگین متحرک 3 ساعته
        df['ma_5'] = df['y'].rolling(5, min_periods=1).mean()      # میانگین متحرک 5 ساعته
        df['roll_std_5'] = df['y'].rolling(5, min_periods=1).std().fillna(0)  # انحراف معیار 5 ساعته
        df['lag_1'] = df['y'].shift(1)  # داده 1 ساعت قبل
        df['lag_2'] = df['y'].shift(2)  # داده 2 ساعت قبل
        df['lag_3'] = df['y'].shift(3)  # داده 3 ساعت قبل
        if isinstance(df.index, pd.DatetimeIndex):
            # تبدیل ساعت به sin/cos برای مدل یادگیری بهتر
            df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
            df['day_of_week'] = df.index.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5,6]).astype(int)
        df = df.dropna()  # حذف ردیف‌های خالی
        return df

    def train_lstm_model(self, X, y, epochs=100, batch_size=32, lr=0.0005):
        """آموزش مدل LSTM"""
        dataset = TimeSeriesDataset(X, y)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)  # ایجاد DataLoader
        model = LSTMModel(input_size=X.shape[2]).to(self.device)  # ساخت مدل LSTM
        criterion = nn.MSELoss()  #   مربعات میانگین تابع خطای MSE
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #رای به‌روزرسانی وزن‌های مدل استفاده می‌کنه Adam optimizer

        for epoch in range(epochs):
            model.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device).unsqueeze(1)  # انتقال به GPU
                optimizer.zero_grad()
                pred = model(xb)          # پیش‌بینی
                loss = criterion(pred, yb)  # محاسبه خطا
                loss.backward()           # backpropagation
                optimizer.step()          # آپدیت وزن‌ها
                losses.append(loss.item())
            if (epoch+1) % 10 == 0: #هر ۱۰ اپوک، مقدار خطا چاپ می‌شود.
                print(f"Epoch {epoch+1}/{epochs}, loss={np.mean(losses):.6f}")
        self.lstm_model = model

    def predict_lstm(self, series, steps=24, sequence_length=30):
        """پیش‌بینی با مدل LSTM"""
        features_df = self.build_features(series)
        series_scaled = self.scaler.transform(features_df.values)  # نرمال‌سازی داده‌ها
        model = self.lstm_model
        model.eval()  #model.eval() می‌گوید که مدل در حالت پیش‌بینی است، نه آموزش
        preds = []

        seq = series_scaled[-sequence_length:].reshape(1, sequence_length, series_scaled.shape[1]) #تغییر شکل داده#
        seq = torch.tensor(seq, dtype=torch.float32).to(self.device)

        with torch.no_grad():  # غیرفعال کردن محاسبات گرادیان
            for _ in range(steps):
                pred = model(seq).cpu().numpy().flatten()[0]
                preds.append(pred)
                next_input = np.append(seq[:,1:,:].cpu().numpy(), [[[pred]*series_scaled.shape[1]]], axis=1) #همه داده‌های قبلی به جز اولین گام حذف می‌شوند
                seq = torch.tensor(next_input, dtype=torch.float32).to(self.device) #اینکار باعث می‌شود مدل گام بعدی را با توجه به پیش‌بینی‌های قبلی بداند

        # inverse_transform پیش‌بینی‌ها را به مقیاس اصلی سری زمانی برمی‌گرداند #  
        # فقط ستون y (مقدار اصلی دما یا سری) را برمی‌گردانیم
        inv_pred = self.scaler.inverse_transform(
            np.hstack([np.array(preds).reshape(-1,1)] + [np.zeros((steps, series_scaled.shape[1]-1))])
        )[:,0]
        return inv_pred

    def predict_future(self, series, steps=24, sequence_length=30):
        """پیش‌بینی با هر سه مدل و گرفتن میانگین"""
        # ARIMA
        arima_pred = ARIMA(series, order=(5,1,0)).fit().forecast(steps)
        # Prophet
        df_prophet = series.reset_index()
        df_prophet.columns = ['ds','y']
        prophet_model = Prophet(daily_seasonality=True)
        prophet_model.fit(df_prophet)
        future = prophet_model.make_future_dataframe(periods=steps, freq='h')
        forecast = prophet_model.predict(future)
        prophet_pred = forecast['yhat'].values[-steps:]
        # LSTM
        lstm_pred = self.predict_lstm(series, steps=steps, sequence_length=sequence_length)
        # Ensemble: میانگین سه مدل
        ensemble_pred = (arima_pred + prophet_pred + lstm_pred) / 3
        return ensemble_pred

# ---------------- Main ----------------
df = pd.read_csv("ml_sensor_data_2000.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)
series = df['temperature'].asfreq('h')  # فرکانس ساعتی

# آماده‌سازی features و scaler
predictor = TimeSeriesPredictor()
features_df = predictor.build_features(series)
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features_df.values)
predictor.scaler = scaler

# آماده‌سازی داده‌ها برای LSTM
sequence_length = 30
X, y_values = [], []
for i in range(len(features_scaled) - sequence_length):
    X.append(features_scaled[i:i+sequence_length])
    y_values.append(features_scaled[i+sequence_length][0])  # ستون y
X = np.array(X)
y_values = np.array(y_values)

# آموزش LSTM
predictor.train_lstm_model(X, y_values, epochs=100, lr=0.0005)

# پیش‌بینی 24 ساعت آینده با ensemble
future_pred = predictor.predict_future(series, steps=24, sequence_length=30)

# محاسبه شاخص‌ها
y_true = series[-24:].values
mae = mean_absolute_error(y_true, future_pred)
rmse = np.sqrt(mean_squared_error(y_true, future_pred))
r2 = r2_score(y_true, future_pred)

print("پیش‌بینی 24 ساعت آینده:", future_pred)
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")

# بارگذاری مدل LSTM از فایل ذخیره شده
checkpoint = torch.load("lstm_model.pth")        # دیکشنری ذخیره شده رو می‌خونه
model = LSTMModel(input_size=checkpoint["input_size"])  # مدل با input_size درست می‌سازه
model.load_state_dict(checkpoint["model_state"])        # وزن‌ها رو به مدل اعمال می‌کنه
model.eval()                                            # حالت پیش‌بینی
