from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('calendar_app.urls')),  # include our calendar_app urls at the root
]
