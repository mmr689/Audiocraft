from IPython import display as ipd
from audiocraft.models import musicgen
from audiocraft.utils.notebook import display_audio
import torch
import torchaudio
import os

# Define the output directory
output_dir = r''

model = musicgen.MusicGen.get_pretrained('medium', device='cuda')
model.set_generation_params(duration=1589)

res = model.generate([
    'A lofi type jazz beat',
], progress=True)

# Save the audio files
for i, audio in enumerate(res):
    audio_cpu = audio.cpu()
    file_path = os.path.join(output_dir, f'audio_{i}.wav')
    torchaudio.save(file_path, audio_cpu, sample_rate=32000)