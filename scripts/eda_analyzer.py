#  backend/analytics/eda_analyzer.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from devices.models import DeviceData

class EDAAnalyzer:
    def __init__(self):
        self.df = self.load_data()
    
    def load_data(self):
        """بارگذاری داده از دیتابیس"""
        data = DeviceData.objects.all().values(
            'device__name',
            'device__device_type',
            'value', 
            'timestamp'
        )
        return pd.DataFrame(data)
    
    def generate_report(self):
        """تولید گزارش کامل EDA"""
        report = {
            'basic_stats': self.basic_statistics(),
            'temporal_patterns': self.temporal_analysis(),
            'correlations': self.correlation_analysis(),
            'outliers': self.find_outliers()
        }
        return report
    
    def create_visualizations(self):
        """ایجاد نمودارهای EDA"""
        self.plot_time_series()
        self.plot_distributions() 
        self.plot_correlation_heatmap()
        self.plot_hourly_patterns()

# استفاده:
analyzer = EDAAnalyzer()
report = analyzer.generate_report()
analyzer.create_visualizations()