from django import forms

class RegisterForm(forms.Form):
    username = forms.CharField(max_length=100, label="User Name", required=True)
    password = forms.CharField(widget=forms.PasswordInput, max_length=100, label="Password", required=False)
    email = forms.EmailField(label="Email", required=False)
    contact = forms.CharField(max_length=20, label="Contact Number", required=False)
