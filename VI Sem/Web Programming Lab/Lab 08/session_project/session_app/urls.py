from django.urls import path
from . import views

urlpatterns = [
    path('', views.first_page, name='first_page'),
    path('second/', views.second_page, name='second_page'),
]
