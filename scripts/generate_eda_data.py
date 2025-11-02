# 📁 backend/scripts/generate_eda_data.py
from data_generator import DataGenerator

# تولید داده زیاد برای EDA
generator = DataGenerator()
generator.generate_historical_data(days=30)  # ۳۰ روز داده
print(" داده کافی برای EDA تولید شد")