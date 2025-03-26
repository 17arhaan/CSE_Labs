from django.shortcuts import render, redirect
from .forms import InsertWorksForm, SearchForm
from .models import Works, Lives

def insert_works(request):
    if request.method == 'POST':
        form = InsertWorksForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('insert_works')
    else:
        form = InsertWorksForm()
    return render(request, 'employee/insert_works.html', {'form': form})

def search_people(request):
    results = []
    if request.method == 'POST':
        form = SearchForm(request.POST)
        if form.is_valid():
            company = form.cleaned_data['company_name']
            works_qs = Works.objects.filter(company_name=company)
            for work in works_qs:
                try:
                    lives = Lives.objects.get(person_name=work.person_name)
                    results.append({'person_name': work.person_name, 'city': lives.city})
                except Lives.DoesNotExist:
                    results.append({'person_name': work.person_name, 'city': 'Unknown'})
    else:
        form = SearchForm()
    return render(request, 'employee/search.html', {'form': form, 'results': results})
