import pandas as pd
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader
import random
import torch
import soundfile as sf

EPS = torch.finfo(torch.float32).eps

class AudioDataset(Dataset):
    def __init__(self, cfg, data_path, train=True, mixed=False):
        self.train = train
        
        self.target_sample_rate = cfg.sample_rate
        self.duration = cfg.duration
        self.device = cfg.device
        self.data_path = data_path
        self.mixed = mixed
        self.df = pd.read_csv(data_path)

    def __len__(self):
        return len(self.df)

    def pre_process(self, audio_path, total_duration):
        duration = self.duration if total_duration >= 3 else total_duration  # Duration is 3 seconds or total_duration if less than 3
        
        if total_duration < self.duration or self.train == False: # 3초보다 짧으면 그냥 사용
            offset = 0.0
        else:
            # 3초보다 길면 랜덤한 구간에서 3초를 가져와서 사용
            max_offset = total_duration - duration  # Calculate the maximum possible offset
            offset = random.uniform(0, max_offset)  # Choose a random offset within the possible range
        
        # Load audio signal file
        wav = AudioSignal(audio_path, offset=offset, duration=duration)
        length = wav.signal_length

        wav.to_mono()
        wav.resample(self.target_sample_rate)

        if wav.duration < self.duration: # 3초보다 짧으면 패딩으로 채우기
          pad_len = int(self.duration * self.target_sample_rate) - wav.signal_length
          wav.zero_pad(0, pad_len)
        assert wav.duration <= self.duration # 3초보다 길면? 안되는데 그러면

        return wav.audio_data, length

    def normalize(self, audio):
        audio = audio/(audio.max(1)[0].abs().unsqueeze(1) + EPS)
        
        rms = (audio**2).mean(1).pow(0.5)
        scalar = 10**(-25/20) / (rms + EPS)
        audio = audio * scalar.unsqueeze(1)
    
        return audio

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        
        audio_path = data['audio_path']
        total_duration = data['duration']
        description = data['caption']
        
        wav, length = self.pre_process(audio_path, total_duration)
        wav = wav.squeeze(1)
        new_wav = wav.clone()
        
        if self.mixed:
            if data['added_audio_path'] != None and isinstance(data['added_audio_path'], str):
                # mix
                new_wav, new_length = self.pre_process(data['added_audio_path'], 3.0)
                new_wav = new_wav.squeeze(1)
                normalized_wav = self.normalize(wav)
                normalized_new_wav = self.normalize(new_wav)
                wav = normalized_wav + normalized_new_wav
                description = data['mixed_caption']
                length = max(length, new_length)

        return wav, description, length


class TestDataset(Dataset):
    def __init__(self, cfg, data_path):
        self.df = pd.read_csv(data_path)[:15]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompts = self.df.iloc[idx]['caption']
        return prompts