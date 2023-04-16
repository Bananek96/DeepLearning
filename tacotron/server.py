from flask import Flask, request, jsonify
import numpy as np
import torch

app = Flask(__name__)


@app.route('/', methods=['GET'])
class Tacotron:
    def __init__(self) -> None:
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.model = self.model.to('cuda')

        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow = self.waveglow.to('cuda')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    def generate_speech(self, text):
        sequences, lengths = self.utils.prepare_input_sequence([text])
        with torch.no_grad():
            mel, _, _ = self.model.infer(sequences, lengths)
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050

        from scipy.io.wavfile import write
        write("audio.wav", rate, audio_numpy)
        with open('audio.wav', 'rb') as f:
            audio_content = f.read()
        return audio_content


@app.route('/', methods=['GET', 'POST'])
def generate_speech():
    if request.method == 'POST':
        text = request.args.get('text')
        if not text:
            return jsonify({'error': 'Missing "text" parameter'}), 400
        model_instance = Tacotron()
        audio_content = model_instance.generate_speech(text)
        return audio_content, 200, {'Content-Type': 'audio/wav', 'Content-Disposition': 'attachment; filename="audio.wav"'}
    else:
        return 'Tacotron 2 server is running'


if __name__ == "__main__":
    app.run(host='localhost', port=5000)
