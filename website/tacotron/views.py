import torch
import librosa.display
import matplotlib.pyplot as plt
import IPython.display as ipd
from django.shortcuts import render


def tacotron2_view(request):
    # Wczytaj model Tacotron2
    sciezka = "D:\github\For_Project\tacotron2_statedict.pt"
    # sciezka = "C:\Users\piter\Desktop\GitHub\ForProject\tacotron2_statedict.pt"
    # sciezka = "C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\tacotron2_statedict.pt"
    model = torch.load(sciezka, map_location=torch.device('cpu'))

    # Wczytaj przykładowy plik dźwiękowy
    sample = 'D:\github\For_Project\waveglow_256channels.pt'
    # sample = 'C:\Users\piter\Desktop\GitHub\ForProject\waveglow_256channels.pt'
    # sample = 'C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\waveglow_256channels.pt'
    audio, sr = librosa.load(sample, sr=22050)

    # Wygeneruj sekwencję mel-spectrogramu za pomocą modelu Tacotron2
    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(audio)

    # Wyświetl wykres mel-spectrogramu
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel_outputs_postnet.cpu().numpy().T, sr=sr, hop_length=256, y_axis='mel', x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.tight_layout()

    # Wyświetl przykładowy plik dźwiękowy
    ipd.display(ipd.Audio(audio, rate=sr))

    # Zwróć widok
    return render(request, 'index.html', context={'alignments': alignments.T})
