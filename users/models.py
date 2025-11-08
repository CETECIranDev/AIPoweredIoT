from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    image = models.ImageField(default='default.jpg', upload_to='profile_pics')
    is_active = models.BooleanField(default=True)
    join_date = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user
