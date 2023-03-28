import json
import os
import sys

import torch
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from ipykernel import kernelapp as app
from tacotron2_code import load_model, inference, convert_mel_to_audio


# Load Tacotron2 model
model = load_model()
model.eval()


@csrf_exempt
def generate_speech(request):
    text_input = request.GET.get('text')
    with torch.no_grad():
        _, mel_outputs, _, _ = inference(model, text_input)
    audio = convert_mel_to_audio(mel_outputs)
    return JsonResponse({'audio': audio})


# Allow the app to be run outside of Jupyter notebooks
if __name__ == '__main__':
    os.environ['DJANGO_SETTINGS_MODULE'] = 'website.settings'
    app.launch_new_instance()
