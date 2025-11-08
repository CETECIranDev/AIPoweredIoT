from celery import shared_task
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync
@shared_task
def broadcast_new_data(device_id, data):
    # sending new data to all connected clients
    channel_layer = get_channel_layer()
    async_to_sync(channel_layer.group_send)(
        f'device_{device_id}',
        {
            'type': 'device_data_update',
            'message': {
                'type': 'new_data',
                'data': data
            }
        }
    )
