from django.shortcuts import render
from django.http import HttpResponse
from django.views import View
import requests
import speech_recognition as sr


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


class SpeechToText(View):
    template_name = 'main/voice.html'

    def get(self, request):
        return render(request, self.template_name)

    def post(self, request):
        if 'record' in request.POST:  # Check if the first button was clicked
            recognizer = sr.Recognizer()
            mic = sr.Microphone()

            with mic as source:
                audio = recognizer.listen(source)

            try:
                text = recognizer.recognize_google(audio)
            except sr.UnknownValueError:
                text = 'Speech recognition could not understand audio'
            except sr.RequestError:
                text = 'Could not request results from Google Speech Recognition service'

            # Render a new template displaying the recognized text
            return render(request, 'main/display_text.html', {'text': text})

        elif 'send' in request.POST:  # Check if the second button was clicked
            text = request.POST.get('text')

            if not text:
                return HttpResponse(status=400, content='Missing "text" parameter')

            # send text to Flask server for further processing if needed
            response = requests.get('http://localhost:5000', params={'text': text})

            # return audio file
            audio_bytes = response.content

            # create a response with the audio data as the content
            response = HttpResponse(content_type='audio/wav')
            response['Content-Disposition'] = 'attachment; filename="audio.wav"'
            response.write(audio_bytes)

            return response

        # If no button was clicked, render the initial form template
        return render(request, self.template_name)