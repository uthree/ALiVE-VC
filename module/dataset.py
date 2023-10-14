import torch
import torchaudio
import glob
from tqdm import tqdm
import os
from module.common import compute_f0


class WaveFileDirectory(torch.utils.data.Dataset):
    def __init__(self, source_dir_paths=[], length=65536, max_files=-1, sampling_rate=22050):
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
    def __init__(self, source_dir_paths=[], length=65536, max_files=-1, sampling_rate=22050):
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


