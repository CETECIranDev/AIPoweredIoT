# 📁 backend/analytics/anomaly_detector.py

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import datetime
from typing import List, Union

# ------------------- مدل‌های تشخیص ناهنجاری -------------------

class MovingAverageDetector:
    """تشخیص ناهنجاری با میانگین متحرک و انحراف معیار"""
    def __init__(self, window: int = 10, threshold: float = 2):
        self.window = window            # تعداد نمونه‌های تاریخی برای محاسبه میانگین و انحراف معیار
        self.threshold = threshold      # ضریب حساسیت (چند برابر انحراف معیار)
        self.history = []               # نگهداری تاریخچه داده‌ها

    def detect(self, data: List[float]) -> List[bool]:
        """داده‌ها را بررسی و لیستی از True/False برای ناهنجاری‌ها برمی‌گرداند"""
        results = []
        for value in data:
            self.history.append(value)
            if len(self.history) > self.window:
                self.history.pop(0)
            mean = np.mean(self.history)
            std = np.std(self.history)
            results.append(abs(value - mean) > self.threshold * std if std > 0 else False)
        return results


class ZScoreDetector:
    """تشخیص ناهنجاری با Z-Score"""
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold      # ضریب حساسیت Z-Score
        self.history = []               # نگهداری تاریخچه داده‌ها

    def detect(self, data: List[float]) -> List[bool]:
        results = []
        for value in data:
            self.history.append(value)
            if len(self.history) < 2:  # اگر داده کافی نباشد نمی‌توان Z-Score محاسبه کرد
                results.append(False)
                continue
            mean = np.mean(self.history)
            std = np.std(self.history)
            z_score = abs(value - mean) / std if std > 0 else 0
            results.append(z_score > self.threshold)
        return results


class IsolationForestDetector:
    """تشخیص ناهنجاری با جنگل ایزوله (Isolation Forest)"""
    def __init__(self, contamination: float = 0.15):
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False         # بررسی اینکه آیا مدل آموزش دیده است یا نه

    def fit(self, X: np.ndarray):
        """آموزش مدل با داده‌های X"""
        self.model.fit(X)
        self.is_trained = True

    def detect(self, X: np.ndarray) -> List[bool]:
        """تشخیص ناهنجاری‌ها"""
        if not self.is_trained:
            self.fit(X)
        predictions = self.model.predict(X)
        return [p == -1 for p in predictions]  # -1 = ناهنجاری، 1 = نرمال


# ------------------- کلاس اصلی Real-Time -------------------

class RealTimeAnomalyDetector:
    """کلاس اصلی برای تشخیص ناهنجاری به صورت Real-Time"""
    def __init__(self):
        self.scaler = StandardScaler()  # برای نرمال‌سازی ویژگی‌ها قبل از مدل‌ها
        self.models = {
            'isolation_forest': IsolationForestDetector(contamination=0.15),
            'z_score': ZScoreDetector(threshold=2.5),
            'moving_avg': MovingAverageDetector(window=10, threshold=2)
        }

    # ------------------- آموزش مدل -------------------
    def train(self, data: Union[pd.DataFrame, List[float]], column_name: str = "temperature"):
        """
        آموزش مدل‌ها
        data: می‌تواند DataFrame یا لیست ساده باشد
        column_name: نام ستونی که برای تشخیص ناهنجاری استفاده می‌شود
        """
        if isinstance(data, pd.DataFrame):
            if column_name not in data.columns:
                raise ValueError(f"Column {column_name} does not exist")

            values = data[column_name].dropna().values
        else:
            values = np.array(data, dtype=float)

        if len(values) == 0:
            raise ValueError("No data available for training")

        # ساخت ویژگی‌ها: مقدار فعلی، ترند، فصلی بودن، میانگین و انحراف معیار ۱۰ نمونه اخیر
        feature_list = []
        for i, val in enumerate(values):
            trend = val - values[i-1] if i > 0 else 0
            seasonality = val - values[i-24] if i >= 24 else 0
            mean = np.mean(values[max(0, i-10):i+1])
            std = np.std(values[max(0, i-10):i+1])
            feature_list.append([val, trend, seasonality, mean, std])

        X = np.array(feature_list)
        X_scaled = self.scaler.fit_transform(X)  # نرمال‌سازی داده‌ها

        # آموزش مدل‌ها و ذخیره با joblib
        for name, model in self.models.items():
            if hasattr(model, "fit"):
                model.fit(X_scaled)
                joblib.dump(model, f"{name}_model.pkl")
        print("models trained successfully")

    # ------------------- پیش‌بینی ناهنجاری -------------------
    def predict(self, sensor_data: List[float]) -> List[dict]:
        """
        تشخیص ناهنجاری در داده‌های سنسور
        خروجی: لیست دیکشنری با index, value, anomaly_score, is_anomaly
        """
        results = []

        for i, value in enumerate(sensor_data):
            try:
                value = float(value)
            except:
                value = 0

            trend = value - sensor_data[i-1] if i > 0 else 0
            seasonality = value - sensor_data[i-24] if i >= 24 else 0
            mean = np.mean(sensor_data[max(0, i-10):i+1])
            std = np.std(sensor_data[max(0, i-10):i+1])

            feature_vector = np.array([[value, trend, seasonality, mean, std]])
            X_scaled = self.scaler.transform(feature_vector)

            # رأی‌گیری بین مدل‌ها برای تصمیم نهایی
            anomaly_votes = []
            for model in self.models.values():
                if hasattr(model, "model"):  # IsolationForest
                    pred = model.model.predict(X_scaled)[0]
                    is_anomaly = 1 if pred == -1 else 0
                else:  # Z-Score یا Moving Average
                    is_anomaly = 1 if model.detect([value])[-1] else 0
                anomaly_votes.append(is_anomaly)

            anomaly_score = np.mean(anomaly_votes)
            is_anomaly = anomaly_score >= 0.34  # حساسیت: 2 مدل از 3 باید ناهنجاری تشخیص دهند

            results.append({
                'index': i,
                'value': value,
                'anomaly_score': round(anomaly_score, 2),
                'is_anomaly': is_anomaly
            })
        print(results)
        return results
       

    # ------------------- ارزیابی مدل -------------------
    def evaluate(self, y_true: List[int], y_pred: List[int]):
        """محاسبه دقت، پرسیژن، ریکال و F1"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


# ------------------- نمونه اجرای تست -------------------
if __name__ == "__main__":
    df = pd.read_csv("ml_sensor_data_2000.csv")
    df['date'] = pd.to_datetime(df['timestamp'])
    df.columns = df.columns.str.strip()

    detector = RealTimeAnomalyDetector()
    detector.train(df, column_name="temperature")  # آموزش مدل‌ها

    sensor_data = df['temperature'].tolist()
    anomalies = detector.predict(sensor_data)       # پیش‌بینی ناهنجاری‌ها

    # تبدیل label به True/False
    y_pred = [a['is_anomaly'] for a in anomalies]
    y_true = df['label'].apply(lambda x: False if str(x).lower() == 'normal' else True).tolist()

    metrics = detector.evaluate(y_true, y_pred)
    print(f"Accuracy:  {metrics['accuracy']*100:.2f}%")
    print(f" Precision: {metrics['precision']*100:.2f}%")
    print(f" Recall:    {metrics['recall']*100:.2f}%")
  
