from django.shortcuts import render
from .models import Institute

def institute_list(request):
    institutes = Institute.objects.all()
    return render(request, 'institutes/institute_list.html', {'institutes': institutes})
