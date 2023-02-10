# Script to Run Inference with a Whisper Model
import whisper 


model = whisper.load_model("tiny")

# Read in Audio File, Pad/Trim it & compute the log_mel_spectrogram
audio = whisper.load_audio("tests/jfk.flac")
audio = whisper.pad_or_trim(audio) #all whisper models were trained on a 30sec window of audio, this is what I want to make smaller

log_mel = whisper.log_mel_spectrogram(audio).to(model.device)


# Run Forward Pass on File (under the hood inside of 'whisper.decode()') w relevant options for decoding generation
options = whisper.DecodingOptions(
        fp16=False, #prevent odd 'slow_conv2d_cpu not implemented for Half' error, as per- https://stackoverflow.com/questions/74725439/runtimeerror-slow-conv2d-cpu-not-implemented-for-half (cant run this op on the CPU for Torch?)
        # temperature=12.0
        beam_size=40, #required parameter if using patience
        # patience=16.0,
)
result = whisper.decode(model, log_mel, options)

print(result)
