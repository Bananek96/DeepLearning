from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.views import View
from pydub import AudioSegment
import io
import requests


class GenerateSpeechView(View):
    template_name = 'main/index.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        if request.method == 'POST':
            text = request.POST.get('text')
            if not text:
                return HttpResponse(status=400, content='Missing "text" parameter')

            # send request to Flask server
            response = requests.get('http://localhost:5000', params={'text': text})

        # return audio file
        audio_bytes = response.content

        # create a response with the audio data as the content
        response = HttpResponse(content_type='audio/wav')
        response['Content-Disposition'] = 'attachment; filename="audio.wav"'
        response.write(audio_bytes)

        return response
