import requests
from django.shortcuts import render


def generate_speech(request):
    text_input = request.GET.get('text')
    url = 'http://localhost:5000'
    payload = {'text': text_input}
    response = requests.get(url, params=payload)
    audio = response.content
    return render(request, 'main/index.html', {'audio': audio})
