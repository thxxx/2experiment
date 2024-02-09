import pandas as pd
from audiotools import AudioSignal
from torch.utils.data import Dataset, DataLoader
import random
import torch
import re
import soundfile as sf
import numpy as np
import pedalboard
import librosa

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
        self.reverb = pedalboard.Reverb()
        self.reverb.room_size = 0.5
        self.reverb.wet_level = 0.9

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
          length=wav.signal_length
          wav.zero_pad(0, pad_len)
        assert wav.duration <= self.duration # 3초보다 길면? 안되는데 그러면
        return wav.audio_data, length

    def normalize(self, audio):
        audio = audio/(audio.max(1)[0].abs().unsqueeze(1) + EPS)
        
        rms = (audio**2).mean(1).pow(0.5)
        scalar = 10**(-25/20) / (rms + EPS)
        audio = audio * scalar.unsqueeze(1)
    
        return audio

    def pitch_augmentation(self, audio_path, total_duration, typed):
        duration = self.duration if total_duration >= 3 else total_duration
        if total_duration < self.duration or self.train == False:
            offset = 0.0
        else:
            max_offset = total_duration - duration
            offset = random.uniform(0, max_offset)
        
        y, sr = librosa.load(audio_path, sr=self.target_sample_rate, offset=offset, duration=duration, mono=True)

        if typed=="low":
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-3)
        if typed=="high":
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=3)
        
        y_shifted = np.expand_dims(y_shifted, axis=0)
        
        length=y_shifted.shape[1]
        if total_duration < self.duration: # 3초보다 짧으면 패딩으로 채우기
          pad_len = int(self.duration * self.target_sample_rate) - y_shifted.shape[1]
          y_shifted = np.pad(y_shifted, pad_width=((0,0), (0, pad_len)), mode='constant', constant_values=0)

        return torch.tensor(y_shifted), length

    def add_reverb(self, audio_path, total_duration):
        duration = self.duration if total_duration >= 3 else total_duration
        if total_duration < self.duration or self.train == False:
            offset = 0.0
        else:
            max_offset = total_duration - duration
            offset = random.uniform(0, max_offset)
        
        y, sr = librosa.load(audio_path, sr=self.target_sample_rate, offset=offset, duration=duration, mono=True)
        y_reverb = self.reverb(y, sample_rate=sr)
        y_reverb = np.expand_dims(y_reverb, axis=0)
        
        length=y_reverb.shape[1]
        if total_duration < self.duration: # 3초보다 짧으면 패딩으로 채우기
          pad_len = int(self.duration * self.target_sample_rate) - y_reverb.shape[1]
          y_reverb = np.pad(y_reverb, pad_width=((0,0), (0, pad_len)), mode='constant', constant_values=0)

        return torch.tensor(y_reverb), length


    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        
        audio_path = data['audio_path']
        total_duration = data['duration']
        description = data['caption']
        
        aug = random.random()
        aug_typed = random.random()

        if not self.train:
            wav, length = self.pre_process(audio_path, total_duration)
            wav = wav.squeeze(1)
            return wav, description, length
    
        if aug<=0.27 and aug>=0.13:
            # print("피치 변경")
            # 피치 변경
            if " low-pitched " in description or " low pitched " in description:
                description = re.sub(" low-pitched ", " high-picthed ", description)
                description = re.sub(" low pitched ", " high picthed ", description)
                wav, length = self.pitch_augmentation(audio_path, total_duration, typed="high")
            elif " high-pitched " in description or " high pitched " in description:
                description = re.sub(" high-pitched ", " low-picthed ", description)
                description = re.sub(" high pitched ", " low picthed ", description)
                wav, length = self.pitch_augmentation(audio_path, total_duration, typed="low")
            else:
                if aug_typed<0.5:
                    wav, length = self.pitch_augmentation(audio_path, total_duration, typed="high")
                    description += ", high pitch"
                else:
                    wav, length = self.pitch_augmentation(audio_path, total_duration, typed="low")
                    description += ", low pitch"
        elif aug>0.4 and aug<0.5:
            # print("리버브 변경")
            wav, length = self.add_reverb(audio_path, total_duration)
            description += ", with reverb"
        else:
            wav, length = self.pre_process(audio_path, total_duration)
            wav = wav.squeeze(1)
    
            if aug<0.5:
                if aug<0.13:
                    # print("볼륨 변경")
                    if aug_typed<0.5:
                        wav *= 2.5
                        description += ", loudly"
                    else:
                        wav /= 2
                        description += ", quietly"
                elif aug>0.27 and aug<0.4:
                    if total_duration>self.duration:
                        # wav = self.normalize(wav)
                        # print("크레센도 변경")
                        how_long_decresendo = random.randint(11,18)/10
                        leng = int(self.target_sample_rate*how_long_decresendo)
                        if aug_typed<0.5:
                            decresend_fade_curve = np.concatenate((np.linspace(1, 0, leng-1000), np.zeros(1000)))
                            wav[0][-leng:] = wav[0][-leng:] * decresend_fade_curve
                            description += ", getting smaller"
                        else:
                            cresend_fade_curve = np.concatenate((np.zeros(1000), np.linspace(0, 1, leng-1000)))
                            wav[0][:leng] = wav[0][:leng] * cresend_fade_curve
                            description += ", getting bigger"
                    

        return wav, description, length


class TestDataset(Dataset):
    def __init__(self, cfg, data_path):
        self.df = pd.read_csv(data_path)[:15]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        prompts = self.df.iloc[idx]['caption']
        return prompts