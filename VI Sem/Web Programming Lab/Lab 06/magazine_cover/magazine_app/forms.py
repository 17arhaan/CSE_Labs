from django import forms

class MagazineCoverForm(forms.Form):
    image = forms.ImageField(required=False, label="Select Cover Image")
    background_color = forms.CharField(max_length=7, required=True, label="Background Color", widget=forms.TextInput(attrs={'type': 'color'}))
    font_color = forms.CharField(max_length=7, required=True, label="Font Color", widget=forms.TextInput(attrs={'type': 'color'}))
    font_size = forms.IntegerField(required=True, min_value=12, max_value=72, label="Font Size (px)")
    font_family = forms.ChoiceField(choices=[('Arial', 'Arial'), ('Times New Roman', 'Times New Roman'), ('Helvetica', 'Helvetica')], label="Font Family")
    title = forms.CharField(max_length=100, required=True, label="Magazine Title")
    tagline = forms.CharField(max_length=150, required=False, label="Tagline")
