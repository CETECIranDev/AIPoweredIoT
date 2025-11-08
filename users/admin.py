from django.contrib import admin
from .models import *
@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'is_active', 'join_date')