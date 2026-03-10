import os
from muspy.datasets import LakhMIDIMatchedDataset

DATA_ROOT = "/pub/hofmann-scratch/students/lziltener/Octuple_MDLM/scratch/datasets/lmd_matched"
os.makedirs(DATA_ROOT, exist_ok=True)

print(f"Downloading LMD-matched to: {DATA_ROOT}")
dataset = LakhMIDIMatchedDataset(root=DATA_ROOT, download_and_extract=True)
dataset.download()
print("Download invoked.")