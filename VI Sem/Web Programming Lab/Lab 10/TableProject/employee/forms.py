from django import forms
from .models import Works

class InsertWorksForm(forms.ModelForm):
    class Meta:
        model = Works
        fields = ['person_name', 'company_name', 'salary']

class SearchForm(forms.Form):
    company_name = forms.CharField(max_length=100, label="Company Name")
