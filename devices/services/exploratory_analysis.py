# نمودار توزیع مقادیر هر سنسور
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



data = pd.read_csv("ml_sensor_data_2000.csv")

# پر کردن مقادیر گم‌شده
data = data.ffill()
# data=data.bfill()


# فقط ستون‌های عددی
numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

# رسم نمودار توزیع برای هر سنسور
# for col in numeric_cols:
#     plt.figure(figsize=(7, 4)) #طول و عرض
#     sns.histplot(data[col], kde=True, bins=30)  #رسم نمودار هیستوگرام(هر مقدار چند بار تکرار شده)   
#                                                 # bins=30 → بازه‌ها رو به ۳۰ قسمت تقسیم کن (برای جزئیات بیشتر)
#                                                 # kde=True → منحنی چگالی احتمالات (Probability Density Curve) هم نشون بده.
#     plt.title(f"Distribution of Sensor Values: {col}")
#     plt.xlabel("value") #افقی
#     plt.ylabel("number of samples ") #عمودی
#     plt.grid(True) #(خط‌های کمکی افقی و عمودی) رو روی نمودار روشن کن
#     plt.show()




# شناسایی الگو های فصلی

#  تبدیل ستون 'timestamp' به نوع داده‌ی datetime
# (یعنی تبدیل رشته‌ها به زمان واقعی قابل درک برای pandas)
data['timestamp'] = pd.to_datetime(data['timestamp'])

# حالا pandas می‌تونه از این ستون ساعت، روز، ماه و سال استخراج کنه

data['hour'] = data['timestamp'].dt.hour         # ساعت ثبت داده
data['dayofweek'] = data['timestamp'].dt.dayofweek  # روز هفته (0=دوشنبه، 6=یک‌شنبه)

#  نمایش چند سطر اول برای بررسی تغییرات
# print(data.head())

# hourly_pattern = data.groupby('hour')['temperature'].mean()
# plt.figure(figsize=(8,4))
# plt.plot(hourly_pattern, marker='o')
# plt.title("Average Temperature by Hour")
# plt.xlabel("Hour of Day")
# plt.ylabel("Average Temperature (°C)")
# plt.grid(True)
# plt.show()



# daily_pattern = data.groupby('dayofweek')['temperature'].mean()
# plt.figure(figsize=(8,4))
# plt.plot(daily_pattern, marker='o', color='orange')
# plt.title("Average Temperature by Day of Week")
# plt.xlabel("Day of Week (0=Mon)")
# plt.ylabel("Average Temperature (°C)")
# plt.grid(True)
# plt.show()


# محاسبه همبستگی بین  سنسور ها

numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns

correlation_matrix = data[numeric_cols].corr()
# print(correlation_matrix)
# خروجی یه جدول مربعی میده که درایه‌های آن بین -1 و 1 هست:
# ۱ → همبستگی کامل مثبت
# -۱ → همبستگی کامل منفی
# ۰ → بدون همبستگی


# plt.figure(figsize=(8,6))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
# plt.title("Correlation between sensors")
# plt.show()
# رنگ آبی → مقادیر پایین/منفی
# رنگ قرمز → مقادیر بالا/مثبت
# خیلی مناسب برای همبستگی چون -1 تا 1 هست.
# رنگ هر خانه میزان همبستگی رو نشون می‌ده،
# عدد داخل خانه مقدار دقیق همبستگی رو نشون می‌ده،
# از آبی (منفی) تا قرمز (مثبت) رنگ‌بندی شده،


#(outlier)با تشخیص داده های پرت IQR


data = data['hour']

# ضریب k
k = 1.5
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - k * IQR
upper_bound = Q3 + k * IQR


outliers = data[(data < lower_bound) | (data > upper_bound)]

# رسم Boxplot با Seaborn
plt.figure(figsize=(8, 6))
sns.boxplot(x=data, color='lightblue')
plt.scatter(outliers.index, outliers, color='red', label='Outliers')
plt.title('Boxplot with Outliers (k=1.5)')
plt.xlabel('Values')
plt.legend()
plt.show()

