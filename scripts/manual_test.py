# 📁 backend/scripts/manual_test.py
from data_generator import DataGenerator

# تست سریع
if __name__ == "__main__":
    gen = DataGenerator()
    
    # فقط ۱۰ داده تستی تولید کن
    for i in range(10):
        for device in gen.devices:
            value = gen.generate_sensor_value(device["type"], i % 24)
            print(f"{device['name']}: {value}")