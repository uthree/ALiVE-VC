import torch
import torchaudio
import glob
from tqdm import tqdm
import os
from module.common import compute_f0


class WaveFileDirectory(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=65536, max_files=-1, sampling_rate=16000):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if max_files != -1:
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class WaveFileDirectoryWithF0(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=65536, max_files=-1, sampling_rate=16000):
        super().__init__()
        print("Loading Data")
        self.path_list = []
        self.data = []
        self.f0 = []
        formats = ["mp3", "wav", "ogg"]
        print("Getting paths")
        for dir_path in source_dir_paths:
            for fmt in formats:
                self.path_list += glob.glob(os.path.join(dir_path, f"**/*.{fmt}"), recursive=True)
        if max_files != -1:
            self.path_list = self.path_list[:max_files]
        print("Chunking")
        for path in tqdm(self.path_list):
            tqdm.write(path)
            wf, sr = torchaudio.load(path) # wf.max() = 1 wf.min() = -1
            # Resample
            wf = torchaudio.functional.resample(wf, sr, sampling_rate)
            # Chunk
            waves = torch.split(wf, length, dim=1)
            tqdm.write(f"    Loading {len(waves)} data...")
            for w in waves:
                if w.shape[1] == length:
                    self.data.append(w[0])
                    self.f0.append(compute_f0(w)[0])
        self.length = length
        print(f"Loaded total {len(self.data)} data.")

    def __getitem__(self, index):
        return self.data[index], self.f0[index]

    def __len__(self):
        return len(self.data)


class ParallelDataset(torch.utils.data.Dataset):
    def __init__(self, dir_path='./parallel'):
        super().__init__()
        print("Loading Data")
        self.src_waves = []
        self.tgt_waves = []
        s_dir = os.path.join(dir_path, "src")
        t_dir = os.path.join(dir_path, "tgt")
        n = 0
        while True:
            path_s = os.path.join(s_dir, f"{n}.wav")
            path_t = os.path.join(t_dir, f"{n}.wav")
            if not (os.path.exists(path_s) and os.path.exists(path_t)):
                break
            wave_s, _ = torchaudio.load(path_s)
            wave_t, _ = torchaudio.load(path_t)
            self.src_waves.append(wave_s.mean(dim=0))
            self.tgt_waves.append(wave_t.mean(dim=0))
            n += 1
        print("Complete!")

    def __getitem__(self, index):
        return self.src_waves[index], self.tgt_waves[index]

    def __len__(self):
        return len(self.src_waves)
