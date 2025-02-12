from django.shortcuts import render

def calendar_view(request):
    return render(request, 'calendar_app/calendar.html')
