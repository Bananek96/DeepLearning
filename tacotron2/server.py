from flask import Flask, request
import torch
from train import load_model
from waveglow.mel2samp import Mel2Samp

app = Flask(__name__)

model = load_model()
model.eval()


@app.route('/', methods=['POST'])
def generate_speech():
    text_input = request.args.get('text')
    with torch.no_grad():
        _, mel_outputs, _, _ = model.inference(text_input)
    audio = Mel2Samp.get_mel(mel_outputs)  # Implement this function to convert mel spectrogram to audio
    return audio


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
