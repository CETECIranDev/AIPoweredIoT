from django.apps import AppConfig
from django.urls import path,include
from .views import DeviceViewSet,DeviceGroupViewSet,device_statistics,AlertViewSet,device_statistics,DeviceDataViewSet,ingest_data
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'devices', DeviceViewSet, basename='device')
router.register(r'device-groups', DeviceGroupViewSet, basename='devicegroup')
router.register(r'alerts', AlertViewSet, basename='alert')
# router.register(r'device-statistics', device_statistics, basename='device-statistics')
# router.register(r'device-data/ingest_data', DeviceDataViewSet, basename='device-data')

urlpatterns = [
    path('api/', include(router.urls)),
    path('device-data/ingest_data/', ingest_data, name='ingest_data'),

]

class DevicesConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'devices'


