from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
import requests

class GenerateSpeechView(View):
    template_name = 'main/index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        text = request.POST.get('text')
        if not text:
            return HttpResponse(status=400, content='Missing "text" parameter')

        # send request to Flask server
        response = requests.get('http://localhost:5000', params={'text': text})
        if response.status_code != 200:
            return HttpResponse(status=500, content='Failed to generate speech')

        # return audio file
        audio_content = response.content
        response = HttpResponse(content_type='audio/wav')
        response['Content-Disposition'] = 'attachment; filename="audio.wav"'
        response.write(audio_content)
        return response
