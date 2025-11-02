# 📁 backend/devices/admin.py
from django.contrib import admin
from .models import Device, DeviceData

@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ['name', 'device_type', 'location', 'created_at']

@admin.register(DeviceData)  
class DeviceDataAdmin(admin.ModelAdmin):
    list_display = ['device', 'value', 'timestamp']