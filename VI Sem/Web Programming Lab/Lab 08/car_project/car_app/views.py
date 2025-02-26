from django.shortcuts import render
from .forms import CarForm

def index(request):
    form = CarForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        manufacturer = form.cleaned_data['manufacturer']
        model = form.cleaned_data['model']
        context = {'manufacturer': manufacturer, 'model': model}
        return render(request, 'car_app/result.html', context)
    return render(request, 'car_app/index.html', {'form': form})
