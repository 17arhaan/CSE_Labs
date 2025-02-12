from django.shortcuts import render
from datetime import date
from .forms import PromotionForm

def promotion_check_view(request):
    result = ""
    if request.method == "POST":
        form = PromotionForm(request.POST)
        if form.is_valid():
            doj = form.cleaned_data['date_of_joining']
            today = date.today()
            years_experience = today.year - doj.year - ((today.month, today.day) < (doj.month, doj.day))
            result = "YES" if years_experience > 5 else "NO"
    else:
        form = PromotionForm()
    context = {'form': form, 'result': result}
    return render(request, 'employee/promotion_form.html', context)
