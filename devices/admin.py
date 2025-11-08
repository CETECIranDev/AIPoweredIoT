# 📁 backend/devices/admin.py
from django.contrib import admin
from .models import *

@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ['name', 'device_type', 'location', 'created_at']

@admin.register(DeviceData)
class DeviceDataAdmin(admin.ModelAdmin):
    list_display = ['device', 'value', 'timestamp']



@admin.register(DeviceGroup)
class DeviceGroupAdmin(admin.ModelAdmin):
    list_display = ('name', 'created_by')

@admin.register(Alert)
class AlertAdmin(admin.ModelAdmin):
    list_display = ('device', 'severity', 'is_resolved', 'created_at')