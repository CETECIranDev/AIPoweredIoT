from rest_framework import authentication, permissions
from rest_framework.exceptions import AuthenticationFailed
from devices.models import Device
from rest_framework_simplejwt.authentication import JWTAuthentication
from rest_framework import exceptions


# JWT Authentication for users
class UserJWTAuthentication(JWTAuthentication):
    def authenticate(self, request):
        # توکن JWT را بررسی می‌کند
        header = self.get_header(request)
        if header is None:
            return None
        raw_token = self.get_raw_token(header)
        validated_token = self.get_validated_token(raw_token)
        return self.get_user(validated_token), validated_token



# Device API Key Authentication
class DeviceAuthentication(authentication.BaseAuthentication):
    """احراز هویت دستگاه‌ها با API Key"""
    def authenticate(self, request):
        api_key = request.META.get('HTTP_X_API_KEY')
        if not api_key:
            return None
        try:
            device = Device.objects.get(api_key=api_key, is_active=True)
            return (device, None)
        except Device.DoesNotExist:
            raise AuthenticationFailed('Invalid API key')



class IsDeviceOwner(permissions.BasePermission):
    """دسترسی فقط برای مالک دستگاه"""
    def has_object_permission(self, request, view, obj):
        return obj.owner == request.user


class IsAdminOrReadOnly(permissions.BasePermission):
    """فقط admin می‌تواند تغییر دهد، بقیه فقط GET"""
    def has_permission(self, request, view):
        if request.method in permissions.SAFE_METHODS:
            return True
        return request.user and request.user.is_staff
