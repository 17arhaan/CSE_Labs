from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='home'),
    path('metadata/', views.metadata, name='metadata'),
    path('reviews/', views.reviews, name='reviews'),
    path('publisher/', views.publisher, name='publisher'),
]
