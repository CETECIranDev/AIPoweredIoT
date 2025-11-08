from django.shortcuts import render
from rest_framework.views import APIView
from users.authentication import DeviceAuthentication,IsDeviceOwner
from users.throttling import DeviceDataThrottle

class DeviceDataView(APIView):
    authentication_classes = [DeviceAuthentication]

class DeviceDetailView(APIView):
    permission_classes = [IsDeviceOwner]

class DeviceDataView(APIView):
    throttle_classes = [DeviceDataThrottle]
