from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status, viewsets
from .models import *
from .serializer import *
from django.shortcuts import get_object_or_404
from django.db.models import Avg, Count
from datetime import timedelta
from django.utils import timezone
from rest_framework.decorators import api_view, permission_classes, throttle_classes
from users.authentication import DeviceAuthentication
from users.throttling import DeviceDataThrottle

#----- Device Management (CRUD) -----
class DeviceViewSet(viewsets.ModelViewSet):
    queryset = Device.objects.all()
    serializer_class = DeviceSerializer
    permission_classes = [IsAuthenticated]


#----- Reports and Statistics -----
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def device_statistics(request, device_id):
    try:
        device = get_object_or_404(Device, id=device_id)
        stats = {
            'total_data_points': device.devicedata_set.count(),
            'last_24h_avg': device.devicedata_set.filter(timestamp__gte=timezone.now() - timedelta(hours=24)).aggregate(
                Avg('value'))['value__avg'],
            'anomaly_count': device.devicedata_set.filter().count()
        }
        return Response(stats)
    except Exception as e:
        return Response({'error': str(e)})


#----- Grouping Devices -----
class DeviceGroupViewSet(viewsets.ModelViewSet):
    serializer_class = DeviceGroupSerializer

    def get_queryset(self):
        return DeviceGroup.objects.filter(created_by=self.request.user)

    def perform_create(self, serializer):
        serializer.save(created_by=self.request.user)


#----- Alert Management -----
class AlertViewSet(viewsets.ModelViewSet):
    queryset = Alert.objects.all()
    serializer_class = AlertSerializer
    permission_classes = [IsAuthenticated]


class DeviceDataViewSet(viewsets.ModelViewSet):
    queryset = DeviceData.objects.all()
    serializer_class = DeviceDataSerializer
    permission_classes = [IsAuthenticated]

@api_view(['POST'])
@permission_classes([DeviceAuthentication])
@throttle_classes([DeviceDataThrottle])
def ingest_data(request):
    """
    دریافت داده از سنسورها (IoT devices)
    """
    # داده‌های ارسال‌شده توسط سنسور
    additional_data = request.data.get('additional_data', {})
    device_name = additional_data.get('device_name')
    value = request.data.get('value')

    if not device_name or value is None:
        return Response({'error': 'device_name و value الزامی هستند'}, status=status.HTTP_400_BAD_REQUEST)

    # بررسی وجود دستگاه در دیتابیس
    try:
        device = Device.objects.get(name=device_name)
    except Device.DoesNotExist:
        return Response({'error': f'Device "{device_name}" not found'}, status=status.HTTP_404_NOT_FOUND)

    # ذخیره داده در DeviceData
    DeviceData.objects.create(device=device, value=value)

    return Response({
        'message': 'Data stored successfully!',
        'device': device.name,
        'value': value
    }, status=status.HTTP_201_CREATED)