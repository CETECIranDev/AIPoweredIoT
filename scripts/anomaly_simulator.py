# 📁 backend/scripts/anomaly_simulator.py
import random
import time
import requests
from datetime import datetime, timedelta


class AnomalySimulator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
    
    def simulate_temperature_spike(self, duration_minutes=5):
        """شبیه‌سازی افزایش ناگهانی دما"""
        print(" شبیه‌سازی افزایش ناگهانی دما...")
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)
        
        while datetime.now() < end_time:
            # افزایش سریع دما از ۲۵ به ۵۰ درجه
            progress = (datetime.now() - start_time).total_seconds() / (duration_minutes * 60)
            temperature = 25 + (25 * progress) + random.uniform(-2, 2)
            
            payload = {
                "api_key": "temp_sensor_1",
                "value": round(temperature, 2),
                "additional_data": {
                    "anomaly_type": "temperature_spike",
                    "simulated": True
                }
            }
            
            requests.post(f"{self.base_url}/api/device-data/ingest_data/", json=payload)
            print(f" دمای anomalous: {temperature:.1f}°C")
            time.sleep(10)  # هر ۱۰ ثانیه
    
    def simulate_sensor_failure(self):
        """شبیه‌سازی خرابی سنسور"""
        print(" شبیه‌سازی خرابی سنسور...")
        
        for i in range(10):
            # ارسال مقدار ثابت (نشانه خرابی)
            payload = {
                "api_key": "humidity_sensor_1", 
                "value": 0,
                "additional_data": {
                    "anomaly_type": "sensor_failure",
                    "simulated": True
                }
            }
            
            requests.post(f"{self.base_url}/api/device-data/ingest_data/", json=payload)
            print(" سنسور خراب: مقدار = ۰")
            time.sleep(30)

if __name__ == "__main__":
    simulator = AnomalySimulator()
    
    print(" شبیه‌سازی Anomaly:")
    simulator.simulate_temperature_spike(duration_minutes=3)
    time.sleep(2)
    simulator.simulate_sensor_failure()