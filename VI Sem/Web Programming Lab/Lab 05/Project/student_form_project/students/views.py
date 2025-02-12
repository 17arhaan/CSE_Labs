from django.shortcuts import render
from .forms import StudentForm

def student_form_view(request):
    form = StudentForm()
    last_percentage = None
    
    if 'student_entries' not in request.session:
        request.session['student_entries'] = []
    entries = request.session.get('student_entries')
    
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():

            name = form.cleaned_data['name']
            date_of_birth = form.cleaned_data['date_of_birth']
            address = form.cleaned_data['address']
            contact_number = form.cleaned_data['contact_number']
            email = form.cleaned_data['email']
            english_marks = form.cleaned_data['english_marks']
            physics_marks = form.cleaned_data['physics_marks']
            chemistry_marks = form.cleaned_data['chemistry_marks']

            total_marks = english_marks + physics_marks + chemistry_marks
            percentage = (total_marks / 300) * 100
            last_percentage = round(percentage, 2)

            entry = (
                f"Name: {name}\n"
                f"DOB: {date_of_birth}\n"
                f"Address: {address}\n"
                f"Contact: {contact_number}\n"
                f"Email: {email}\n"
                f"English: {english_marks}, Physics: {physics_marks}, Chemistry: {chemistry_marks}\n"
                f"Percentage: {last_percentage}%\n"
                "----------------------\n"
            )

            entries.append(entry)
            request.session['student_entries'] = entries
            form = StudentForm()
    
    context = {
        'form': form,
        'entries': ''.join(entries),
        'last_percentage': last_percentage,
    }
    
    return render(request, 'students/student_form.html', context)
