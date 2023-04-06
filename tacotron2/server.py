from flask import Flask, request
import matplotlib.pylab as plt
from hparams import create_hparams
from train import load_model
from text import text_to_sequence
from waveglow.denoiser import Denoiser
import numpy as np
import torch
import IPython.display as ipd
import sys

sys.path.append('waveglow/')

hparams = create_hparams()
hparams.sampling_rate = 22050

checkpoint_path = "D:/github/For_Project/tacotron2_statedict.pt"
# checkpoint_path = "C:\Users\piter\Desktop\GitHub\ForProject\tacotron2_statedict.pt"
# checkpoint_path = "C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\tacotron2_statedict.pt"

model = load_model(hparams)
model.load_state_dict(torch.load(checkpoint_path)['state_dict'])
_ = model.cuda().eval().half()

waveglow_path = 'D:/github/For_Project/waveglow_256channels_universal_v5.pt'
# waveglow_path = 'C:\Users\piter\Desktop\GitHub\ForProject\waveglow_256channels.pt'
# waveglow_path = 'C:\Users\huber\Desktop\0STUDIA\Projekt_Neuronówa\For_Project\waveglow_256channels.pt'

waveglow = torch.load(waveglow_path)['model']
waveglow.cuda().eval().half()
for k in waveglow.convinv:
    k.float()
denoiser = Denoiser(waveglow)


app = Flask(__name__)


def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none')


@app.route('/', methods=['GET'])
def generate_speech():

    text_input = request.args.get('text')
    sequence = np.array(text_to_sequence(text_input, ['english_cleaners']))[None, :]
    sequence = torch.autograd.Variable(
        torch.from_numpy(sequence)).cuda().long()

    mel_outputs, mel_outputs_postnet, _, alignments = model.inference(sequence)
    # plot_data((mel_outputs.float().data.cpu().numpy()[0],
    #            mel_outputs_postnet.float().data.cpu().numpy()[0],
    #            alignments.float().data.cpu().numpy()[0].T))

    with torch.no_grad():
        audio = waveglow.infer(mel_outputs_postnet, sigma=0.666)
    ipd.Audio(audio[0].data.cpu().numpy(), rate=hparams.sampling_rate)

    audio_denoised = denoiser(audio, strength=0.01)[:, 0]
    ipd.Audio(audio_denoised.cpu().numpy(), rate=hparams.sampling_rate)
    return audio


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
