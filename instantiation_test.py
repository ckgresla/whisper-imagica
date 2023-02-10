# See if I can instantiate a smaller whisper model -- it looks like we cAN! 

import whisper 

from whisper import ModelDimensions
from whisper.model import Whisper

# original dims
# ModelDimensions(n_mels=80, n_audio_ctx=1500, n_audio_state=384, n_audio_head=6, n_audio_layer=4, n_vocab=51865, n_text_ctx=448, n_text_state=384, n_text_head=6, n_text_layer=4)

# playing w new dims -- presumably can change and load in a custom model w same codes
md = ModelDimensions(n_mels=80,
                     n_audio_ctx=1500,
                     n_audio_state=384,
                     n_audio_head=1,
                     n_audio_layer=9,
                     n_vocab=51865,
                     n_text_ctx=448,
                     n_text_state=384,
                     n_text_head=12,
                     n_text_layer=4
)

smol = Whisper(md)

print(smol)

# test out a forward pass w nothingness
audio = whisper.load_audio("samples/jfk.wav")
audio = whisper.pad_or_trim(audio)
log_mel = whisper.log_mel_spectrogram(audio).to(smol.device)

options = whisper.DecodingOptions(fp16=False)

out = whisper.decode(smol, log_mel, options)
print(out)
