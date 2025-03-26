from django.shortcuts import render, redirect
from .forms import CategoryForm, PageForm
from .models import Category, Page

def add_category(request):
    if request.method == 'POST':
        form = CategoryForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('display')
    else:
        form = CategoryForm()
    return render(request, 'directory/add_category.html', {'form': form})

def add_page(request):
    if request.method == 'POST':
        form = PageForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('display')
    else:
        form = PageForm()
    return render(request, 'directory/add_page.html', {'form': form})

def display(request):
    categories = Category.objects.all()
    pages = Page.objects.all()
    return render(request, 'directory/display.html', {'categories': categories, 'pages': pages})
