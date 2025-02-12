from django.urls import path
from . import views

urlpatterns = [
    path('', views.student_form_view, name='student_form'),
]
