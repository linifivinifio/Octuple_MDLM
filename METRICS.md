# Evaluation Metrics Documentation

## Overview

This document describes all evaluation metrics used for symbolic music generation, their meanings, and expected ranges.

---

## Unconditional Generation Metrics

Compares generated samples against training data distribution to assess realism and diversity.

### Distribution Similarity Metrics

#### **Pitch Class Histogram KL Divergence** (`pch_kl`)

- **What**: KL divergence between pitch class (0-11, C to B) distributions
- **Meaning**: How closely generated pitch usage matches training data
- **Range**: [0, ∞), lower is better
- **Good Value**: < 0.5 (very realistic), < 1.0 (acceptable)
- **Poor Value**: > 2.0 (unrealistic pitch distribution)

#### **Duration KL Divergence** (`duration_kl`)

- **What**: KL divergence between note duration distributions
- **Meaning**: How closely generated note lengths match training data
- **Range**: [0, ∞), lower is better
- **Good Value**: < 0.5 (realistic rhythms)
- **Poor Value**: > 2.0 (unrealistic note lengths, e.g., 196-hour notes)

#### **Velocity KL Divergence** (`velocity_kl`)

- **What**: KL divergence between velocity (dynamics) distributions
- **Meaning**: How closely generated dynamics match training data
- **Range**: [0, ∞), lower is better
- **Good Value**: < 0.5

#### **Note Density KL Divergence** (`note_density_kl`)

- **What**: KL divergence between notes-per-bar distributions
- **Meaning**: How closely generated note counts match training data
- **Range**: [0, ∞), lower is better
- **Good Value**: < 0.3
- **Poor Value**: > 1.0 (too sparse or too dense)

---

### Musical Coherence Metrics

#### **Self-Similarity** (`self_similarity`)

- **What**: Average cosine similarity between consecutive 4-bar pitch/rhythm vectors
- **Meaning**: Measures internal musical structure and repetition
- **Range**: [0, 1], higher is better
- **Good Value**: 0.3-0.7 (has structure without being repetitive)
- **Poor Value**: < 0.1 (random noise), > 0.9 (completely repetitive)

#### **Pitch Range Mean** (`pitch_range_mean`)

- **What**: Average span between lowest and highest pitch per sample
- **Meaning**: Musical range coverage
- **Range**: [0, 127] semitones
- **Typical**: 24-48 (2-4 octaves for melody), 60-84 (5-7 octaves for trio)

#### **Average Polyphony** (`avg_polyphony`)

- **What**: Mean number of simultaneous notes
- **Meaning**: Harmonic density
- **Range**: [1, ∞)
- **Typical**: 1.0-1.5 (melody), 2.0-4.0 (trio)

---

### Diversity Metrics

#### **Sample Diversity** (`sample_diversity`)

- **What**: Average pairwise L2 distance between samples in feature space
- **Meaning**: How varied the generated samples are
- **Range**: [0, ∞), higher is better
- **Good Value**: > 0.3 (diverse outputs)
- **Poor Value**: < 0.1 (mode collapse, all samples similar)

---

### Validity Metrics

#### **Valid Samples Percentage** (`valid_samples_pct`)

- **What**: Percentage of samples that decode to valid MIDI
- **Meaning**: Model generates structurally valid music
- **Range**: [0, 100]%
- **Good Value**: > 95%
- **Poor Value**: < 80%

---

## Infilling Metrics

Evaluates reconstruction accuracy and musical quality in masked regions.

### Reconstruction Accuracy (Error Metrics)

#### **Pitch Accuracy** (`pitch_accuracy`)

- **What**: % of pitches matching ground truth in masked region
- **Meaning**: How accurately the model reconstructs the original pitches
- **Range**: [0, 100]%, higher is better
- **Good Value**: > 40% (difficult task)
- **Excellent Value**: > 60%

#### **Duration Accuracy** (`duration_accuracy`)

- **What**: % of durations matching ground truth in masked region
- **Meaning**: How accurately the model reconstructs rhythm
- **Range**: [0, 100]%, higher is better
- **Good Value**: > 30%

#### **Token Accuracy** (`token_accuracy`)

- **What**: % of full 8-tuple tokens exactly matching ground truth
- **Meaning**: Perfect reconstruction rate (strictest metric)
- **Range**: [0, 100]%, higher is better
- **Good Value**: > 10% (very strict)
- **Note**: Low values expected since all 8 attributes must match

---

### Boundary Coherence Metrics

#### **Boundary Pitch Smoothness** (`boundary_pitch_smoothness`)

- **What**: Average pitch difference at mask boundaries (bars 16 & 32)
- **Meaning**: How well infilled music connects to context
- **Range**: [0, 127] semitones, lower is better
- **Good Value**: < 5 (smooth transitions)
- **Poor Value**: > 12 (jarring jumps)

#### **Boundary Rhythm Smoothness** (`boundary_rhythm_smoothness`)

- **What**: Duration similarity at mask boundaries
- **Meaning**: Rhythmic continuity at edges
- **Range**: [0, ∞), lower is better
- **Good Value**: < 2.0

---

### Musical Quality in Masked Region

#### **Infilled PCH KL** (`infilled_pch_kl`)

- **What**: KL divergence between infilled and original pitch distributions
- **Meaning**: Tonal consistency (same key/mode)
- **Range**: [0, ∞), lower is better
- **Good Value**: < 0.5

#### **Infilled Note Density Error** (`infilled_density_error`)

- **What**: Absolute difference in notes-per-bar between infilled and original
- **Meaning**: Activity level consistency
- **Range**: [0, ∞), lower is better
- **Good Value**: < 1.0 note/bar

---

## Metric Availability by Task

| Metric | Unconditional | Infilling |
| ------ | ------------- | --------- |
| PCH KL | ✓ (vs dataset) | ✓ (vs original) |
| Duration KL | ✓ | ✓ |
| Velocity KL | ✓ | ✓ |
| Note Density KL | ✓ | - |
| Self-Similarity | ✓ | ✓ |
| Pitch Range | ✓ | ✓ |
| Sample Diversity | ✓ | ✓ |
| Valid Samples % | ✓ | ✓ |
| Pitch Accuracy | - | ✓ |
| Duration Accuracy | - | ✓ |
| Token Accuracy | - | ✓ |
| Boundary Smoothness | - | ✓ |
| Infilled Density Error | - | ✓ |

---

## Interpretation Guide

### Early Training (Steps 0-1000)

- Expect **poor** distribution KL (> 2.0)
- Low accuracy (< 10%)
- Poor self-similarity (< 0.2)
- Many invalid samples

### Mid Training (Steps 1000-5000)

- Distribution KL improving (1.0-2.0)
- Accuracy rising (20-40%)
- Self-similarity emerging (0.3-0.5)
- Mostly valid samples (> 90%)

### Well-Trained (Steps 5000+)

- Good distribution KL (< 0.5)
- Decent accuracy (> 40%)
- Musical structure (0.4-0.6)
- Nearly all valid (> 95%)

---

## Usage

```bash
# Evaluate unconditional generation
python -m smdiff.cli.evaluate \
  --task uncond \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --dataset_id pop909_trio_octuple \
  --n_samples 100

# Evaluate infilling
python -m smdiff.cli.evaluate \
  --task infill \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --dataset_id pop909_trio_octuple \
  --mask_start_bar 16 \
  --mask_end_bar 32 \
  --n_samples 100
```
