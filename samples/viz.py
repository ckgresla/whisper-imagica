# Visualizing a Mel spectrogram of a WAV file
# Taken from- https://github.com/musikalkemist/AudioSignalProcessingForML/blob/master/18%20-%20Extracting%20Mel%20Spectrograms%20with%20Python/Extracting%20Mel%20Spectrograms.ipynb
import librosa
import librosa.display
import IPython.display as ipd
import matplotlib.pyplot as plt


scale_file = "jfk.wav"
scale_file = "jfk.flac"
ipd.Audio(scale_file)

# load audio files with librosa
scale, sr = librosa.load(scale_file) #get a sampling rate from file


# Get Mel Filter Banks
filter_banks = librosa.filters.mel(n_fft=2048, sr=16_000, n_mels=80) #sampling rate here is user defined?
print(f"Filter Banks Shape: {filter_banks.shape}")

# plt.figure(figsize=(25, 10))
# librosa.display.specshow(filter_banks, sr=sr, x_axis="linear")
# plt.colorbar(format="%+2.f")
# plt.title("Filter Banks")
# plt.show()


# Extracting the Mel spectrogram & visualizing
mel_spectrogram = librosa.feature.melspectrogram(scale, sr=sr, n_fft=2048, n_mels=80)
print(f"Mel spectrogram Shape: {mel_spectrogram.shape}")

log_mel_spectrogram = librosa.power_to_db(mel_spectrogram)
print(f"Log-Mel spectrogram Shape: {log_mel_spectrogram.shape}")
log_mel_spectrogram = mel_spectrogram


plt.figure(figsize=(25, 10))
librosa.display.specshow(log_mel_spectrogram, x_axis="time", y_axis="mel", sr=sr)
plt.colorbar(format="%+2.f")
plt.show()
