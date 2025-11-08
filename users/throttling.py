from rest_framework.throttling import UserRateThrottle

class DeviceDataThrottle(UserRateThrottle):
    """Rate limiting برای دریافت داده‌ها از دستگاه‌ها"""
    scope = 'device_data'
    rate = '100/hour'
