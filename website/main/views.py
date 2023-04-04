import requests
from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
import torch



class Tacotron2API(APIView):
    def post(self, request):
        text = request.data.get('text')
        model_path = '/../../tacotron2/model.py'
        synthesizer = Synthesizer(model_path)
        speech = synthesizer.synthesize(text)
        return Response({'speech': speech})


def generate_speech(request):
    text_input = request.GET.get('text')
    url = 'http://localhost:8888/tree/interference.py'  # Replace with your Jupyter Notebook URL
    payload = {'text': text_input}
    response = requests.get(url, params=payload)
    audio = response.content
    return render(request, 'main/index.html', {'audio': audio})
