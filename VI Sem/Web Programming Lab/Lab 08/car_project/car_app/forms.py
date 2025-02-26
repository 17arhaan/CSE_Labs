from django import forms

class CarForm(forms.Form):
    manufacturer = forms.ChoiceField(
        choices=[
            ('Toyota', 'Toyota'),
            ('Honda', 'Honda'),
            ('Ford', 'Ford'),
            ('BMW', 'BMW'),
            ('Audi', 'Audi')
        ],
        label="Car Manufacturer",
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    model = forms.CharField(
        max_length=100,
        label="Model Name",
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
