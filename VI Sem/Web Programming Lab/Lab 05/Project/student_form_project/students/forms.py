from django import forms

class StudentForm(forms.Form):
    name = forms.CharField(max_length=100, label='Name')
    date_of_birth = forms.DateField(
        widget=forms.DateInput(attrs={'type': 'date'}),
        label='Date of Birth'
    )
    address = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 3}),
        label='Address'
    )
    contact_number = forms.CharField(max_length=15, label='Contact Number')
    email = forms.EmailField(label='Email ID')
    english_marks = forms.IntegerField(min_value=0, max_value=100, label='English Marks')
    physics_marks = forms.IntegerField(min_value=0, max_value=100, label='Physics Marks')
    chemistry_marks = forms.IntegerField(min_value=0, max_value=100, label='Chemistry Marks')
