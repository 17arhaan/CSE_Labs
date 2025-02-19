from django.shortcuts import render
from .forms import MagazineCoverForm

def index(request):
    form = MagazineCoverForm(request.POST or None, request.FILES or None)
    cover = None
    if request.method == 'POST' and form.is_valid():
        cover = form.cleaned_data
    return render(request, 'magazine_app/index.html', {'form': form, 'cover': cover})
