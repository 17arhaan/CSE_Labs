from django.shortcuts import render, redirect

def page1(request):
    if request.method == 'POST':
        name = request.POST.get('name', '')
        marks = request.POST.get('marks', '0')
        request.session['name'] = name
        request.session['marks'] = marks
        return redirect('page2')
    return render(request, 'cgpaapp/page1.html')

def page2(request):
    name = request.session.get('name', 'NoName')
    marks = request.session.get('marks', '0')
    try:
        marks_val = float(marks)
    except ValueError:
        marks_val = 0
    cgpa = marks_val / 50
    return render(request, 'cgpaapp/page2.html', {'name': name, 'cgpa': cgpa})
