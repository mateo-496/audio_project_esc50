import tqdm
import essentia.standard as es
import librosa
import numpy as np
import os
import soundfile as sf

from src.config.config import sample_rate, parameters, cnn_input_length

def data_treatment_training(
    audio_path, 
    n_bands, n_mels, frame_size, hop_size, sample_rate, fft_size
    ):
    labels = []
    log_mel_spectrograms = []
    filenames = os.listdir(audio_path)

    for filename in tqdm.tqdm(filenames, desc="Processing audio files"):

        filename_splitted = filename.split("-")
        label = filename_splitted[-1].split(".")[0]
        label = label.split("_")[0]
        labels.append(int(label))

        file_path = os.path.join(audio_path, filename)

        window = es.Windowing(type="hann")
        spectrum = es.Spectrum(size=fft_size)
        mel = es.MelBands(
            numberBands=n_bands, 
            inputSize=fft_size//2 + 1, 
            sampleRate=sample_rate,
            lowFrequencyBound=0,
            highFrequencyBound=sample_rate / 2
        )

        loader = es.MonoLoader(filename=file_path)
        audio = loader()

        frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
        log_mel_spectrogram = []
        for frame in frames:
            frame_padded = np.pad(frame, (0, fft_size - len(frame)), mode='constant')
            windowed_frame = window(frame_padded)
            spec = spectrum(windowed_frame)
            mel_bands = mel(spec)
            log_mel_spectrogram.append(mel_bands)

        log_mel_spectrogram = np.array(log_mel_spectrogram)
        
        mel_spectrogram_db = 10 * np.log10(log_mel_spectrogram + 1e-10)
        max_db = mel_spectrogram_db.max()
        mel_spectrogram_db = mel_spectrogram_db - max_db
        
        log_mel_spectrograms.append(mel_spectrogram_db)
    return log_mel_spectrograms, np.array(labels)

def data_treatment_testing(
    audio_path, 
    n_bands, n_mels, frame_size, hop_size, sample_rate, fft_size
    ):
    labels = []
    log_mel_spectrograms = []
    filenames = os.listdir(audio_path)

    for filename in tqdm.tqdm(filenames, desc="Processing audio files"):

        file_path = os.path.join(audio_path, filename)

        window = es.Windowing(type="hann")
        spectrum = es.Spectrum(size=fft_size)
        mel = es.MelBands(
            numberBands=n_bands, 
            inputSize=fft_size//2 + 1, 
            sampleRate=sample_rate,
            lowFrequencyBound=0,
            highFrequencyBound=sample_rate / 2
        )

        loader = es.MonoLoader(filename=file_path)
        audio = loader()

        frames = es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size)
        log_mel_spectrogram = []
        for frame in frames:
            frame_padded = np.pad(frame, (0, fft_size - len(frame)), mode='constant')
            windowed_frame = window(frame_padded)
            spec = spectrum(windowed_frame)
            mel_bands = mel(spec)
            log_mel_spectrogram.append(mel_bands)

        log_mel_spectrogram = np.array(log_mel_spectrogram)
        
        mel_spectrogram_db = 10 * np.log10(log_mel_spectrogram + 1e-10)
        max_db = mel_spectrogram_db.max()
        mel_spectrogram_db = mel_spectrogram_db - max_db
        
        log_mel_spectrograms.append(mel_spectrogram_db)
        
    return log_mel_spectrograms

def pad(audio, target_seconds, sample_rate):
    target_len = int(sample_rate * target_seconds)
    n = len(audio)

    if n < target_len:
        audio = np.pad(audio, (0, target_len - n), mode="constant")
    return audio

def time_stretch_augmentation(file_path, sample_rate, rate):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    audio_timestretch = librosa.effects.time_stretch(audio.astype(np.float32), rate=rate)
    return pad(audio_timestretch, 5, sample_rate)

def pitch_shift_augmentation(file_path, sample_rate, semitones):
    audio, _ = librosa.load(file_path, sr=sample_rate)
    return librosa.effects.pitch_shift(audio.astype(np.float32), sr=sample_rate, n_steps=semitones)

def drc_augmentation(file_path, sample_rate, compression):
    if compression == "music_standard":   threshold_db=-20; ratio=2.0; attack_ms=5;  release_ms=50
    elif compression == "film_standard":  threshold_db=-25; ratio=4.0; attack_ms=10; release_ms= 100
    elif compression == "speech":         threshold_db=-18; ratio=3.0; attack_ms=2;  release_ms= 40
    elif compression == "radio":          threshold_db=-15; ratio=3.5; attack_ms=1;  release_ms= 200

    audio, _ = librosa.load(file_path, sr=sample_rate)
    threshold = 10**(threshold_db / 20)

    attack_coeff  = np.exp(-1.0 / (0.001 * attack_ms * sample_rate))
    release_coeff = np.exp(-1.0 / (0.001 * release_ms * sample_rate))
    
    audio_filtered = np.zeros_like(audio)
    gain = 1.0
    
    for n in range(len(audio)):
        abs_audio = abs(audio[n])
        if abs_audio > threshold:
            desired_gain = (threshold / abs_audio) ** (ratio - 1)
        else:
            desired_gain = 1.0
        
        if desired_gain < gain:
            gain = attack_coeff * (gain - desired_gain) + desired_gain
        else:
            gain = release_coeff * (gain - desired_gain) + desired_gain
        
        audio_filtered[n] = audio[n] * gain

    return audio_filtered

def augment_dataset(audio_path, output_path, probability_list):
    filenames = os.listdir(audio_path)

    p1, p2, p3 = probability_list
    os.makedirs(output_path, exist_ok=True)

    for filename in tqdm.tqdm(filenames, desc="Processing audio files"):        
        
        augmentations = []
        audio, _ = librosa.load(os.path.join(audio_path, filename), sr=sample_rate)
        # TS
        if np.random.rand() > p1:
            stretch_rates = [0.81, 0.93, 1.07, 1.23]
            stretch_rate = np.random.choice(stretch_rates)
            audio = time_stretch_augmentation(os.path.join(audio_path, filename), sample_rate, stretch_rate)
            augmentations.append(f"TS{stretch_rate}")
        # PS 
        if np.random.rand() > p2:
            semitones = [-3.5, -2.5, -2, -1, 1, 2.5, 3, 3.5]
            semitone = np.random.choice(semitones)
            audio = pitch_shift_augmentation(os.path.join(audio_path, filename), sample_rate, semitone)
            augmentations.append(f"PS{semitone}")

        # DRC
        if np.random.rand() > p3:
            compressions = ["radio", "film_standard", "music_standard", "speech"]
            compression = np.random.choice(compressions)
            audio = drc_augmentation(os.path.join(audio_path, filename), sample_rate, compression)
            augmentations.append(f"DRC{compression}")

        for aug in augmentations:
            filename_splitted = filename.split(".")
            filename = filename_splitted[0] + f"_{aug}." + filename_splitted[-1]
        sf.write(os.path.join(output_path, filename), audio, 44100)

def create_augmented_datasets(input_path, output_path):
    probability_lists = [
        [0.0 , 1.0, 1.0],
        [1.0 , 1.0, 0.0],
        [1.0 , 0.0, 1.0],
        [0.0 , 0.0, 0.0],
        [0.5 , 0.5, 0.5]]
    for i, probability_list in enumerate(probability_lists):
        augmented_path = os.path.join(output_path, f"{i+1}")
        os.makedirs(augmented_path, exist_ok=True)
        augment_dataset(input_path, augmented_path, probability_list)

def create_log_mel(input_path, output_path):
    directories = os.listdir(input_path)
    X, y = [], []

    for directory in directories:
        log_mels, labels = data_treatment_training(os.path.join(input_path, directory), **parameters)
        X.extend(log_mels)
        y.extend(labels)
    
    X_array = np.empty(len(X), dtype=object)
    for i, spec in enumerate(X):
        X_array[i] = spec

    y = np.array(y)
    os.makedirs(output_path, exist_ok=True)
    
    np.save(os.path.join(output_path, "X.npy"), X_array, allow_pickle=True)
    np.save(os.path.join(output_path, 'y.npy'), y)
    return X, y

if __name__ == "__main__":
    input_path = "data/audio/0"
    output_base_path = "data/audio"

    create_augmented_datasets(input_path, output_base_path)

