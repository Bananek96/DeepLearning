import IPython.display as ipd
import matplotlib.pyplot as plt
import numpy as np
import torch
from django.http import HttpResponse
from django.shortcuts import render

from django_for_jupyter import init_django
from hparams import create_hparams
from layers import STFT
from audio_processing import griffin_lim
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')


def generate_audio(request):
    init_django("website")  # assuming 'website' is the name of your Django project
    hparams = create_hparams()
    hparams.sampling_rate = 22050
    
    checkpoint_path = "D:\github\For_Project\tacotron2_statedict.pt"
    # checkpoint_path = "C:\Users\piter\Desktop\GitHub\ForProject\tacotron2_statedict.pt"
    # checkpoint_path = "C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\tacotron2_statedict.pt"

    waveglow_path = 'D:\github\For_Project\waveglow_256channels.pt'
    # waveglow_path = 'C:\Users\piter\Desktop\GitHub\ForProject\waveglow_256channels.pt'
    # waveglow_path = 'C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\waveglow_256channels.pt'

    model = load_model(hparams)
    model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
    model = model.cuda().eval().half()

    waveglow = torch.load(waveglow_path)['model']
    waveglow.cuda().eval().half()
    for k in waveglow.convinv:
        k.float()
    denoiser = Denoiser(waveglow)

    if request.method == 'POST':
        text = request.POST['text']
        sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
        sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
        with torch.no_grad():
            audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
            audio_denoised = denoiser(audio, strength=0.01)[:, 0]
        response = HttpResponse(content_type='audio/wav')
        griffin_audio = griffin_lim(mel_outputs_postnet.float().data.cpu().numpy()[0], stft_fn=STFT())
        ipd.Audio(griffin_audio, rate=hparams.sampling_rate)
        response.write(audio_denoised.cpu().numpy().tobytes())
        return response

    return render(request, 'index.html')
