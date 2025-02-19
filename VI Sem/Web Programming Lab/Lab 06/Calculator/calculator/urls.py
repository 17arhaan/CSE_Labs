from django.urls import path
from .views import arithmetic_view

urlpatterns = [
    path('', arithmetic_view, name='arithmetic'),
]
