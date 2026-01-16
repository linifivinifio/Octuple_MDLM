"""Octuple data module - PyTorch dataset and MIDI encoding for octuple representation."""
from .dataset import OctupleDataset
from .encoding import OctupleEncoding

__all__ = ['OctupleDataset', 'OctupleEncoding']
