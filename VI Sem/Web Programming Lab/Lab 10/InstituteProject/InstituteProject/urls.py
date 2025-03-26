from django.urls import path
from .views import institute_list

urlpatterns = [
    path('', institute_list, name='institute_list'),
]
