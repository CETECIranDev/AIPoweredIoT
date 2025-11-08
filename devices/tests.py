from django.test import TestCase
from rest_framework.test import APITestCase
from django.contrib.auth.models import User
from .models import Device

class DeviceAPITest(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='1234')
        self.client.login(username='test', password='1234')
        self.device = Device.objects.create(name='Device1', device_type='Sensor', location='Lab')

    def test_get_devices(self):
        response = self.client.get('/api/devices/')
        self.assertEqual(response.status_code, 200)

    def test_create_device(self):
        data = {'name': 'Device2', 'device_type': 'Camera', 'location': 'Room'}
        response = self.client.post('/api/devices/', data)
        self.assertEqual(response.status_code, 201)
