#!/usr/bin/env python
# coding: utf-8
"""Local-only event-dataset loaders. These classes never download data."""
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
import tonic.transforms as transforms
from tonic.io import read_mnist_file

def _require_dir(path, dataset_name):
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{dataset_name} local dataset folder was not found:\\n  {path}\\nThis artifact is local-data-only and will not download datasets.")
    if not path.is_dir():
        raise NotADirectoryError(f"Expected a directory, got: {path}")
    return path

NMNIST_DTYPE=np.dtype([('x',int),('y',int),('t',int),('p',int)])
class LocalNMNIST(Dataset):
    sensor_size=(34,34,2)
    def __init__(self, root, T, denoise_filter_time=10000):
        self.root=_require_dir(root,'N-MNIST'); self.T=int(T)
        files=sorted(self.root.glob('**/*.bin'))
        if not files: raise RuntimeError(f"No N-MNIST .bin files were found under {self.root}.")
        self.data=[]; self.targets=[]
        for path in files:
            try: label=int(path.parent.name)
            except ValueError as exc: raise ValueError(f"N-MNIST expects class directories named 0..9; could not infer label from {path}.") from exc
            self.data.append(path); self.targets.append(label)
        self.transform=transforms.Compose([transforms.Denoise(filter_time=int(denoise_filter_time)), transforms.ToFrame(sensor_size=self.sensor_size,n_time_bins=self.T)])
        print(f"[LOCAL DATA] N-MNIST | root={self.root} | samples={len(self.data)} | T={self.T}", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self,index):
        events=read_mnist_file(str(self.data[int(index)]),dtype=NMNIST_DTYPE)
        return self.transform(events), int(self.targets[int(index)])

DVSGESTURE_DTYPE=np.dtype([('x',np.int16),('y',np.int16),('p',np.bool_),('t',np.int64)])
class LocalDVSGesture(Dataset):
    sensor_size=(128,128,2)
    def __init__(self, root, T, denoise_filter_time=10000):
        self.root=_require_dir(root,'DVS128 Gesture'); self.T=int(T)
        files=sorted(self.root.glob('**/*.npy'))
        if not files: raise RuntimeError(f"No DVS Gesture .npy files were found under {self.root}.")
        self.data=[]; self.targets=[]
        for path in files:
            try: label=int(path.stem)
            except ValueError as exc: raise ValueError(f"DVS Gesture expects integer .npy filenames; got {path.name!r}.") from exc
            self.data.append(path); self.targets.append(label)
        self.transform=transforms.Compose([transforms.Denoise(filter_time=int(denoise_filter_time)), transforms.ToFrame(sensor_size=self.sensor_size,n_time_bins=self.T)])
        print(f"[LOCAL DATA] DVS128 Gesture | root={self.root} | samples={len(self.data)} | T={self.T}", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self,index):
        idx=int(index); arr=np.asarray(np.load(self.data[idx]))
        if arr.ndim!=2 or arr.shape[1]<4: raise ValueError(f"Unexpected DVS Gesture array shape {arr.shape} in {self.data[idx]}.")
        events=np.empty(arr.shape[0],dtype=DVSGESTURE_DTYPE)
        events['x']=arr[:,0].astype(np.int16,copy=False); events['y']=arr[:,1].astype(np.int16,copy=False); events['p']=arr[:,2].astype(np.bool_,copy=False)
        events['t']=np.asarray(np.rint(arr[:,3].astype(np.float64)*1000.0),dtype=np.int64)
        return self.transform(events), int(self.targets[idx])

NCALTECH_DTYPE=np.dtype([('x',int),('y',int),('t',int),('p',int)])
class LocalNCaltech101(Dataset):
    native_sensor_size=(240,180,2); target_size=(64,64); polarity_channels=2
    def __init__(self, root, T, denoise_filter_time=10000):
        self.root=_require_dir(root,'N-Caltech101'); self.T=int(T)
        files=sorted(self.root.glob('**/*.bin'))
        if not files: raise RuntimeError(f"No N-Caltech101 .bin files were found under {self.root}.")
        self.data=list(files); self.targets=[p.parent.name for p in files]
        self.transform=transforms.Compose([transforms.Denoise(filter_time=int(denoise_filter_time)), transforms.Downsample(sensor_size=self.native_sensor_size,target_size=self.target_size), transforms.ToFrame(sensor_size=(64,64,2),n_time_bins=self.T)])
        print(f"[LOCAL DATA] N-Caltech101 | root={self.root} | samples={len(self.data)} | classes={len(set(self.targets))} | T={self.T}", flush=True)
    def __len__(self): return len(self.data)
    def __getitem__(self,index):
        idx=int(index); events=read_mnist_file(str(self.data[idx]),dtype=NCALTECH_DTYPE)
        if len(events)==0: raise RuntimeError(f"No events read from {self.data[idx]}.")
        events['x']-=events['x'].min(); events['y']-=events['y'].min()
        return self.transform(events), self.targets[idx]
