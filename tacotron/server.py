import numpy as np
import torch
from flask import Flask, request, Response

app = Flask(__name__)


class Tacotron:
    def __init__(self) -> None:
        self.model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
        self.model = self.model.to('cuda')

        self.waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
        self.waveglow = self.waveglow.remove_weightnorm(self.waveglow)
        self.waveglow = self.waveglow.to('cuda')
        self.utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')

    def generate(self, text):
        sequences, lengths = self.utils.prepare_input_sequence([text])
        with torch.no_grad():
            mel, _, _ = self.model.infer(sequences, lengths)
            audio = self.waveglow.infer(mel)
        audio_numpy = audio[0].data.cpu().numpy()
        rate = 22050

        from scipy.io.wavfile import write
        write("audio.wav", rate, audio_numpy)

        # return audio_numpy

        # return the audio data as a byte string instead of writing to file
        from io import BytesIO
        with BytesIO() as output:
            write(output, rate, audio_numpy)
            audio_bytes = output.getvalue()

        return audio_bytes


@app.route('/', methods=['GET', "POST"])
def generate_speech():
    text = request.args.get('text')
    model_instance = Tacotron()
    audio = model_instance.generate(text)

    return Response(audio, mimetype='audio/wav')


if __name__ == '__main__':
    app.run(host='localhost', port=5000)
