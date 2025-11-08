# devices/routing.py
from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r'ws/device/(?P<device_id>\d+)/$', consumers.DeviceDataConsumer.as_asgi()),
]
