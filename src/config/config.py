import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

@dataclass(frozen=True)
class ProcessingConfig:
    audio_path: Path = Path("data/audio/0")
    augmented_path: Path = Path("data/audio/")
    log_mel_path: Path = Path("data/preprocessed")
    n_bands: int = 128
    n_mels: int = 128
    frame_size: int = 1024
    hop_size: int = 1024
    sample_rate: int = 44100
    fft_size: int = 8192
    target_seconds: float = 5.0
    augmentation_probability_lists = [
            [0.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0], [0.5, 0.5, 0.5],
        ]
    time_stretch_rates = [0.81, 0.93, 1.07, 1.23]
    pitch_shift_rates = [-3.5, -2.5, -2, -1, 1, 2.5, 3, 3.5]
    drc_types = ["radio", "filmstandard", "musicstandard", "speech"]

@dataclass(frozen=True)
class DatasetConfig:
    cnn_input_length: int = 128
    sample_rate: int = 44100
    esc50_labels: List[str] = field(default_factory=lambda: [
        'dog', 'rooster', 'pig', 'cow', 'frog',
        'cat', 'hen', 'insects', 'sheep', 'crow',
        'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
        'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
        'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
        'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
        'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
        'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
        'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
        'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
    ])

@dataclass(frozen=True)
class DownloadConfig:
    repo_url: str = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
    repo_dst_dir: Path = Path("data")
    audio_dst_dir: Path = field(init=False)
    extracted_dir: Path = os.path.join(repo_dst_dir, "ESC-50-master")
    audio_src_dir = os.path.join(extracted_dir, "audio")
    paths_to_delete: List[str] = field(default_factory=lambda: [
        ".gitignore", "esc50.gif", "LICENSE", "pytest.ini", "README.md",
        "requirements.txt", "tests", "meta", ".github", ".circleci", "ESC-50-master"
    ])

    def __post_init__(self):
        object.__setattr__(self, "audio_dst_dir", self.repo_dst_dir / "audio" / "0")



parameters = {
    "n_bands"  : 128,
    "n_mels" : 128,
    "frame_size" : 1024,
    "hop_size": 1024,
    "sample_rate": 44100,
    "fft_size": 8192,
}

cnn_input_length = 128

sample_rate = 44100

esc50_labels = [
    'dog', 'rooster', 'pig', 'cow', 'frog',
    'cat', 'hen', 'insects', 'sheep', 'crow',
    'rain', 'sea_waves', 'crackling_fire', 'crickets', 'chirping_birds',
    'water_drops', 'wind', 'pouring_water', 'toilet_flush', 'thunderstorm',
    'crying_baby', 'sneezing', 'clapping', 'breathing', 'coughing',
    'footsteps', 'laughing', 'brushing_teeth', 'snoring', 'drinking_sipping',
    'door_wood_knock', 'mouse_click', 'keyboard_typing', 'door_wood_creaks', 'can_opening',
    'washing_machine', 'vacuum_cleaner', 'clock_alarm', 'clock_tick', 'glass_breaking',
    'helicopter', 'chainsaw', 'siren', 'car_horn', 'engine',
    'train', 'church_bells', 'airplane', 'fireworks', 'hand_saw'
]

# download.py
repo_url = "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip"
repo_dst_dir = "data"
audio_dst_dir = os.path.join(repo_dst_dir, "audio", "0")

paths_to_delete = [
    ".gitignore",
    "esc50.gif",
    "LICENSE",
    "pytest.ini",
    "README.md",
    "requirements.txt",
    "tests",
    "meta",
    ".github",
    ".circleci"
]
