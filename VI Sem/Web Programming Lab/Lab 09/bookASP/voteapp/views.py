from django.shortcuts import render

VOTE_COUNTS = {'good': 0, 'satisfactory': 0, 'bad': 0}

def vote_view(request):
    total = sum(VOTE_COUNTS.values())
    percentages = {'good': 0, 'satisfactory': 0, 'bad': 0}
    if total > 0:
        percentages['good'] = round(VOTE_COUNTS['good'] * 100 / total)
        percentages['satisfactory'] = round(VOTE_COUNTS['satisfactory'] * 100 / total)
        percentages['bad'] = round(VOTE_COUNTS['bad'] * 100 / total)
    if request.method == 'POST':
        choice = request.POST.get('vote')
        if choice in VOTE_COUNTS:
            VOTE_COUNTS[choice] += 1
        total = sum(VOTE_COUNTS.values())
        if total > 0:
            percentages['good'] = round(VOTE_COUNTS['good'] * 100 / total)
            percentages['satisfactory'] = round(VOTE_COUNTS['satisfactory'] * 100 / total)
            percentages['bad'] = round(VOTE_COUNTS['bad'] * 100 / total)
    return render(request, 'voteapp/vote.html', {'percentages': percentages})
