from rest_framework import serializers
from .models import Device,DeviceData,DeviceGroup,Alert



class DeviceSerializer(serializers.ModelSerializer):
    class Meta:
        model = Device
        fields = '__all__'

class DeviceDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceData
        fields = '__all__'

class DeviceGroupSerializer(serializers.ModelSerializer):
    class Meta:
        model = DeviceGroup
        fields = '__all__'

class AlertSerializer(serializers.ModelSerializer):
    class Meta:
        model = Alert
        fields = '__all__'

