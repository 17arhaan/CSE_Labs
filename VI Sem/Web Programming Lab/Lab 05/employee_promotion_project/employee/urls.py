from django.urls import path
from .views import promotion_check_view

urlpatterns = [
    path('', promotion_check_view, name='promotion_check'),
]
