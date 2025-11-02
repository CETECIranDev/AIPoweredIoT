# 📁 backend/scripts/data_generator.py
import random
import time
from datetime import datetime, timedelta
import requests
import json

class DataGenerator:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.devices = [
            {"api_key": "temp_sensor_1", "type": "temperature", "name": "دماسنج اتاق A"},
            {"api_key": "humidity_sensor_1", "type": "humidity", "name": "رطوبت‌سنج اتاق A"},
            {"api_key": "pressure_sensor_1", "type": "pressure", "name": "فشارسنج آزمایشگاه"},
        ]
    
    def generate_sensor_value(self, sensor_type, hour):
        """تولید مقدار سنسور بر اساس نوع و ساعت روز"""
        if sensor_type == "temperature":
            # دمای بین ۲۰ تا ۳۰ درجه با تغییرات روزانه
            base_temp = 25
            daily_variation = 5 * math.sin(2 * math.pi * hour / 24)
            noise = random.uniform(-1, 1)
            return round(base_temp + daily_variation + noise, 2)
        
        elif sensor_type == "humidity":
            # رطوبت بین ۵۰٪ تا ۸۰٪
            base_humidity = 65
            daily_variation = 10 * math.sin(2 * math.pi * (hour - 6) / 24)
            noise = random.uniform(-2, 2)
            return round(base_humidity + daily_variation + noise, 2)
        
        elif sensor_type == "pressure":
            # فشار بین ۱۰۱۰ تا ۱۰۱۶ هکتوپاسکال
            return round(1013 + random.uniform(-3, 3), 2)
    
    def send_sensor_data(self, device, value):
        """ارسال داده به سرور"""
        payload = {
            "api_key": device["api_key"],
            "value": value,
            "additional_data": {
                "device_name": device["name"],
                "device_type": device["type"],
                "timestamp": datetime.now().isoformat()
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/device-data/ingest_data/",
                json=payload,
                timeout=5
            )
            if response.status_code == 201:
                print(f" داده ارسال شد: {device['name']} = {value}")
            else:
                print(f" خطا: {response.status_code}")
        except Exception as e:
            print(f" خطای شبکه: {e}")
    
    def generate_historical_data(self, days=7):
        """تولید داده تاریخی برای ۷ روز گذشته"""
        print(f" تولید داده تاریخی برای {days} روز گذشته...")
        
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        current_time = start_time
        while current_time <= end_time:
            hour = current_time.hour
            
            for device in self.devices:
                value = self.generate_sensor_value(device["type"], hour)
                self.send_sensor_data(device, value)
            
            # به جلو برو در زمان
            current_time += timedelta(hours=1)
            time.sleep(0.1)  # تأثیر کمی
    
    def start_real_time_simulation(self, interval=30):
        """شروع شبیه‌سازی داده بلادرنگ"""
        print(f" شروع شبیه‌سازی بلادرنگ (هر {interval} ثانیه)")
        
        try:
            while True:
                current_hour = datetime.now().hour
                
                for device in self.devices:
                    value = self.generate_sensor_value(device["type"], current_hour)
                    self.send_sensor_data(device, value)
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print(" شبیه‌سازی متوقف شد")

if __name__ == "__main__":
    import math
    
    generator = DataGenerator()
    
    print(" انتخاب گزینه:")
    print("1. تولید داده تاریخی (۷ روز)")
    print("2. شبیه‌سازی بلادرنگ")
    print("3. هر دو")
    
    choice = input("گزینه مورد نظر (1/2/3): ")
    
    if choice in ["1", "3"]:
        generator.generate_historical_data(days=7)
    
    if choice in ["2", "3"]:
        generator.start_real_time_simulation(interval=30)