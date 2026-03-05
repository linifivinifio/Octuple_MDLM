import os
from muspy.datasets import LakhMIDIMatchedDataset

# Ensure the full path exists before MusPy tries to use it
os.makedirs("datasets/lmd_matched", exist_ok=True)

dataset = LakhMIDIMatchedDataset(root="datasets/lmd_matched", download_and_extract=True)
dataset.download()   # will attempt to fetch the archive