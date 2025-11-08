from django.db import models
from django.contrib.auth.models import User



class Device(models.Model):
    name = models.CharField(max_length=100)
    device_type = models.CharField(max_length=50)
    location = models.CharField(max_length=100)
    api_key = models.CharField(max_length=100, unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class DeviceData(models.Model):
    device = models.ForeignKey(Device, on_delete=models.CASCADE)
    value = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.device.name} - {self.value}"


class DeviceGroup(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField()
    devices = models.ManyToManyField(Device)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE)

    def __str__(self):
        return self.name


class Alert(models.Model):
    SEVERITY_CHOICES = [('low', 'Low'), ('medium', 'Medium'), ('high', 'High')]
    device = models.ForeignKey(Device, on_delete=models.CASCADE)
    message = models.TextField()
    severity = models.CharField(max_length=10, choices=SEVERITY_CHOICES)
    is_resolved = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)

