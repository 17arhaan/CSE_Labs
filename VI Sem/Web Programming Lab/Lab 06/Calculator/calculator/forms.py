from django import forms

OPERATION_CHOICES = [
    ('add', 'Addition'),
    ('subtract', 'Subtraction'),
    ('multiply', 'Multiplication'),
    ('divide', 'Division'),
]

class ArithmeticForm(forms.Form):
    number1 = forms.IntegerField(label="Number 1")
    number2 = forms.IntegerField(label="Number 2")
    operation = forms.ChoiceField(choices=OPERATION_CHOICES, label="Operation")
