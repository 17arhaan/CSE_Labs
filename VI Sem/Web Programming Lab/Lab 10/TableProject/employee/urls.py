from django.urls import path
from .views import insert_works, search_people

urlpatterns = [
    path('insert/', insert_works, name='insert_works'),
    path('search/', search_people, name='search_people'),
]
