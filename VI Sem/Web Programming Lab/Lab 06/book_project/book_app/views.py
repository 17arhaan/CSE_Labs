from django.shortcuts import render

def index(request):
    return render(request, 'book_app/index.html')

def metadata(request):
    return render(request, 'book_app/metadata.html')

def reviews(request):
    return render(request, 'book_app/reviews.html')

def publisher(request):
    return render(request, 'book_app/publisher.html')
