import requests
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse


def index(request):
    return HttpResponse("Hello world!")


def generate_speech(request):
    text_input = request.GET.get('text')
    url = 'http://localhost:8888/tree'  # Replace with your Jupyter Notebook URL
    payload = {'text': text_input}
    response = requests.get(url, params=payload)
    audio = response.content
    return render(request, 'main/index.html', {'audio': audio})
