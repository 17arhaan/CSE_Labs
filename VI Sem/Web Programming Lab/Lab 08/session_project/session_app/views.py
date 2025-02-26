from django.shortcuts import render, redirect

def first_page(request):
    if request.method == 'POST':
        name = request.POST.get('name')
        roll = request.POST.get('roll')
        subject = request.POST.get('subject')
        request.session['name'] = name
        request.session['roll'] = roll
        request.session['subject'] = subject
        return redirect('second_page')
    return render(request, 'session_app/firstPage.html')

def second_page(request):
    name = request.session.get('name', 'N/A')
    roll = request.session.get('roll', 'N/A')
    subject = request.session.get('subject', 'N/A')
    context = {
        'name': name,
        'roll': roll,
        'subject': subject,
    }
    return render(request, 'session_app/secondPage.html', context)
