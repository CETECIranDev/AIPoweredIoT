# """
# ASGI config for AIPoweredIoTProject project.
#
# It exposes the ASGI callable as a module-level variable named ``application``.
#
# For more information on this file, see
# https://docs.djangoproject.com/en/5.2/howto/deployment/asgi/
# """
#
# import os
#
# from django.core.asgi import get_asgi_application
#
# os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AIPoweredIoTProject.settings')
#
# application = get_asgi_application()

import os
from channels.auth import AuthMiddlewareStack
from django.core.asgi import get_asgi_application
import devices.routing
from channels.routing import ProtocolTypeRouter, URLRouter
import devices.routing

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AIPoweredIoT.settings')



application = ProtocolTypeRouter({
    "http": get_asgi_application(),
    "websocket": AuthMiddlewareStack(
        # URLRouter(
        #     devices.routing.websocket_urlpatterns
        # )
    ),
})
