# Evaluation System Quick Start

## Files Created

- **METRICS.md** - Comprehensive documentation of all metrics, their meanings, and good value ranges
- **src/smdiff/metrics/** - Metrics computation library
  - `common.py` - Shared utilities (KL divergence, histograms, etc.)
  - `unconditional.py` - Unconditional generation metrics
  - `infilling.py` - Infilling-specific metrics
- **src/smdiff/cli/evaluate.py** - Main evaluation CLI

## Usage

### Evaluate Unconditional Generation

```bash
python -m smdiff.cli.evaluate \
  --task uncond \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --dataset_id pop909_trio_octuple \
  --n_samples 100 \
  --save_samples
```

### Evaluate Infilling

```bash
python -m smdiff.cli.evaluate \
  --task infill \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --dataset_id pop909_trio_octuple \
  --mask_start_bar 16 \
  --mask_end_bar 32 \
  --n_samples 100 \
  --save_samples
```

### Evaluate Existing Samples

```bash
python -m smdiff.cli.evaluate \
  --task uncond \
  --sample_dir runs/octuple_ddpm_trio_octuple/samples/uncond \
  --dataset_id pop909_trio_octuple \
  --n_samples 100
```

## Outputs

- Console: Formatted metrics with descriptions
- `evaluation/{task}/metrics_{task}.json` - JSON file with all metrics
- `evaluation/{task}/samples/` - Generated samples (if --save_samples used)

## Key Metrics by Task

### Unconditional
- **Distribution KL** (< 0.5 = good): PCH, Duration, Velocity, Note Density
- **Self-Similarity** (0.3-0.7 = good): Musical structure
- **Sample Diversity** (> 0.3 = good): Varied outputs
- **Valid Samples** (> 95% = good): Structurally valid

### Infilling
- **Pitch/Duration Accuracy** (> 40% = good): Reconstruction quality
- **Boundary Smoothness** (< 5 semitones = good): Coherent transitions
- **Infilled PCH KL** (< 0.5 = good): Tonal consistency
- All unconditional metrics for overall quality

## See METRICS.md for Complete Documentation
