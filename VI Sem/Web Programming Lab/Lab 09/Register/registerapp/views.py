from django.shortcuts import render
from .forms import RegisterForm

def register(request):
    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            email = form.cleaned_data.get('email')
            contact = form.cleaned_data.get('contact')
            return render(request, 'registerapp/success.html', {
                'username': username,
                'email': email,
                'contact': contact,
            })
    else:
        form = RegisterForm()
    return render(request, 'registerapp/register.html', {'form': form})
