from django.shortcuts import render
from .forms import ArithmeticForm

def arithmetic_view(request):
    result = None
    if request.method == 'POST':
        form = ArithmeticForm(request.POST)
        if form.is_valid():
            num1 = form.cleaned_data['number1']
            num2 = form.cleaned_data['number2']
            operation = form.cleaned_data['operation']
            if operation == 'add':
                result = num1 + num2
            elif operation == 'subtract':
                result = num1 - num2
            elif operation == 'multiply':
                result = num1 * num2
            elif operation == 'divide':
                result = num1 / num2 if num2 != 0 else 'Error: Division by zero'
    else:
        form = ArithmeticForm()
    context = {'form': form, 'result': result}
    return render(request, 'calculator/index.html', context)
