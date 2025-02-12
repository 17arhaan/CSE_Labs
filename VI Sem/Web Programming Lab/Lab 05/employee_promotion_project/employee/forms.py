from django import forms

EMPLOYEE_CHOICES = [
    ('emp1', 'Employee 1'),
    ('emp2', 'Employee 2'),
    ('emp3', 'Employee 3'),
]

class PromotionForm(forms.Form):
    employee_id = forms.ChoiceField(choices=EMPLOYEE_CHOICES, label="Employee ID")
    date_of_joining = forms.DateField(widget=forms.DateInput(attrs={'type': 'date'}), label="Date of Joining")
