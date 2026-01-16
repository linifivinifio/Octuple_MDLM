# Symbolic Music Discrete Diffusion

A symbolic music generation framework based on absorbing state diffusion, supporting both grid-based (OneHot) and event-based (Octuple) MIDI representations.

This implementation provides flexible tools for training, sampling, and evaluating discrete diffusion models on symbolic music data, with support for unconditional generation and infilling tasks.

---

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Data Preparation](#data-preparation)
- [Training Models](#training-models)
- [Sampling & Evaluation](#sampling--evaluation)
- [Available Components](#available-components)
- [CLI Reference](#cli-reference)
- [Attribution](#attribution)

---

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/symbolic-music-discrete-diffusion.git
cd symbolic-music-discrete-diffusion

# Create conda environment
conda env create -f env.yml
conda activate beatmaster

# Or use venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

---

## Quick Start

```bash
# 1. Prepare data (POP909 dataset)
python -m smdiff.cli.prepare_data \
    --tokenizer_id trio_octuple \

# 2. Train model
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple \
    --bars 64 \
    --batch_size 16

# 3. Generate samples
python -m smdiff.cli.sample \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 8

# 4. Evaluate model
python -m smdiff.cli.evaluate \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 100
```

---

## Data Preparation

The `prepare_data.py` script converts MIDI files into tokenized `.npy` caches for training.

### Usage

```bash
python -m smdiff.cli.prepare_data \
    --root_dir <midi_directory> \
    --tokenizer_id <tokenizer> \
    --bars <num_bars> \
    --cache_path <output.npy> \
    [--limit <max_files>] \
    [--num_workers <n>]
```

### Arguments

- `--root_dir`: Directory containing MIDI files (searched recursively)
- `--tokenizer_id`: Tokenization format (see [Available Tokenizers](#available-tokenizers))
- `--bars`: Number of bars per sequence (typically 64)
- `--cache_path`: Output path for `.npy` cache
- `--limit`: Process only N files (default: all)
- `--num_workers`: Parallel workers (default: CPU count)
- `--max_tensors_per_ns`: Max sequences per MIDI (default: 5)
- `--strict_tempo`: Reject files with tempo changes

### Examples

**Prepare trio (3-track) octuple dataset:**
```bash
python -m smdiff.cli.prepare_data \
    --root_dir data/POP909 \
    --tokenizer_id trio_octuple \
    --bars 64 \
    --cache_path data/POP909_trio_octuple.npy \
    --num_workers 8
```

**Prepare melody-only grid dataset:**
```bash
python -m smdiff.cli.prepare_data \
    --root_dir data/POP909 \
    --tokenizer_id melody \
    --bars 64 \
    --cache_path data/POP909_melody.npy
```

**Quick test (100 files):**
```bash
python -m smdiff.cli.prepare_data \
    --root_dir data/POP909 \
    --tokenizer_id trio_octuple \
    --bars 64 \
    --cache_path data/POP909_trio_octuple_test.npy \
    --limit 100
```

### Output

Creates a `.npy` file containing tokenized sequences. After preparation, register the dataset in `src/smdiff/data/registry.py`:

```python
register_dataset(
    "pop909_trio_octuple",
    DatasetSpec(
        dataset_path="data/POP909_trio_octuple.npy",
        tracks="trio",
        bars=64,
        notes=1024,  # Variable length for octuple
        tokenizer_id="trio_octuple"
    )
)
```

---

## Training Models

The `train.py` script trains diffusion models on prepared datasets.

### Argument Precedence

Configuration is loaded in this order (later overrides earlier):

1. **Default config** (`src/smdiff/configs/<model>.yaml`)
2. **Dataset config** (via `--dataset_id`, applies dataset-specific settings)
3. **Command-line arguments** (highest priority)

### Usage

```bash
python train.py \
    --model <model_id> \
    --dataset_id <dataset_id> \
    [--bars <n>] \
    [--batch_size <n>] \
    [--lr <rate>] \
    [--steps <n>]
```

### Key Arguments

- `--model`: Model architecture (see [Available Models](#available-models))
- `--dataset_id`: Dataset registry ID
- `--bars`: Sequence length in bars (overrides config)
- `--batch_size`: Training batch size
- `--lr`: Learning rate
- `--steps`: Total training steps
- `--log_dir`: Output directory (default: `runs/<model>_<dataset>`)
- `--ema`: Enable EMA (default: True)
- `--ema_rate`: EMA decay rate (default: 0.9999)

### Examples

**Train octuple DDPM on trio data:**
```bash
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple \
    --batch_size 16 \
    --lr 1e-4 \
    --steps 100000
```

**Train transformer VAE on melody:**
```bash
python train.py \
    --model schmu_tx_vae \
    --dataset_id pop909_melody \
    --batch_size 32 \
    --steps 50000
```

**Resume training from checkpoint:**
```bash
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple \
    --log_dir runs/octuple_ddpm_trio_octuple \
    --resume
```

**Override config with CLI args:**
```bash
# Dataset config sets batch_size=16, we override to 32
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple \
    --batch_size 32  # CLI takes precedence
```

### Output Structure

```
runs/<model>_<dataset>/
├── checkpoints/
│   ├── best.pt              # Best validation checkpoint
│   ├── ema_best.pt          # Best EMA checkpoint
│   ├── <sampler>_<step>.th # Regular checkpoints
│   └── <sampler>_ema_<step>.th
├── configs/
│   └── config.yaml          # Merged config used for training
├── samples/                 # Generated during training
│   └── *.mid
└── tensorboard/             # Training logs
```

### Converting Training Samples

Samples generated during training are saved as `.npy` arrays. Convert to MIDI:

```bash
python npz_to_midi.py \
    --input runs/octuple_ddpm_trio_octuple/samples/sample_step_10000.npy \
    --tokenizer_id trio_octuple \
    --output_dir converted_samples/
```

**Batch conversion:**
```bash
for f in runs/octuple_ddpm_trio_octuple/samples/*.npy; do
    python npz_to_midi.py \
        --input "$f" \
        --tokenizer_id trio_octuple \
        --output_dir converted_samples/
done
```

---

## Sampling & Evaluation

### Sampling with `sample.py`

Generate new music or perform infilling with trained models.

#### Unconditional Generation

```bash
python -m smdiff.cli.sample \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 16 \
    --sample_steps 100 \
    --ema
```

**Arguments:**
- `--task`: `uncond` or `infill`
- `--model`: Model ID from registry
- `--load_dir`: Path to trained model
- `--dataset_id`: Dataset ID (for tokenizer/config)
- `--n_samples`: Number of samples to generate
- `--sample_steps`: Diffusion steps (0 = use config default)
- `--load_step`: Checkpoint step (0 = best checkpoint)
- `--ema` / `--no-ema`: Use EMA weights (default: True)

#### Infilling (Single MIDI)

```bash
python -m smdiff.cli.sample \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi data/POP909/001/001.mid \
    --mask_start_bar 16 \
    --mask_end_bar 32 \
    --n_samples 4
```

**Arguments:**
- `--input_midi`: Single MIDI file for conditioning
- `--mask_start_bar`: Start of masked region
- `--mask_end_bar`: End of masked region

#### Infilling (Multiple MIDIs)

Generate multiple samples from a directory of MIDIs:

```bash
python -m smdiff.cli.sample \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi_dir data/POP909 \
    --samples_per_midi 4 \
    --mask_start_bar 16 \
    --mask_end_bar 32
```

**Arguments:**
- `--input_midi_dir`: Directory of MIDI files (all `.mid` files used)
- `--samples_per_midi`: Samples per conditioning MIDI

### Evaluation with `evaluate.py`

Compute task-specific metrics on generated samples. See `METRICS.md` for detailed metric documentation.

#### Unconditional Evaluation (Generate)

```bash
python -m smdiff.cli.evaluate \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 100 \
    --sample_steps 100 \
    --save_samples
```

**Metrics computed:**
- Distribution KL divergences (pitch, duration, velocity, note density)
- Musical coherence (self-similarity, pitch range, polyphony)
- Sample diversity and validity

#### Unconditional Evaluation (Load Existing)

```bash
python -m smdiff.cli.evaluate \
    --task uncond \
    --sample_dir runs/octuple_ddpm_trio_octuple/samples/uncond \
    --dataset_id pop909_trio_octuple \
    --n_samples 100
```

#### Infilling Evaluation (Generate)

```bash
python -m smdiff.cli.evaluate \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi_dir data/POP909 \
    --samples_per_midi 4 \
    --n_samples 32 \
    --mask_start_bar 16 \
    --mask_end_bar 32
```

**Metrics computed:**
- Reconstruction accuracy (pitch, duration, token)
- Boundary coherence (pitch/rhythm smoothness at mask edges)
- Infilled region quality (distribution similarity, note density)

#### Output

Metrics saved to `runs/<model>/metrics/metrics_<task>.json`:

```json
{
  "pch_kl": 0.15,
  "duration_kl": 0.23,
  "self_similarity": 0.75,
  "sample_diversity": 45.2,
  "valid_samples_pct": 98.5
}
```

---

## Available Components

### Available Models

Registered in `src/smdiff/registry/models.py`:

| Model ID | Architecture | Description |
|----------|-------------|-------------|
| `octuple_ddpm` | Absorbing Diffusion | DDPM for octuple (event-based) encoding |
| `schmu_tx_vae` | Transformer VAE | Grid-based transformer VAE |
| `schmu_conv_vae` | Convolutional VAE | Grid-based conv VAE |

### Available Tokenizers

Registered in `src/smdiff/tokenizers/registry.py`:

| Tokenizer ID | Type | Tracks | Description |
|--------------|------|--------|-------------|
| `melody` | Grid (OneHot) | 1 | Melody-only, 1024 steps (64 bars × 16 steps) |
| `trio` | Grid (OneHot) | 3 | Piano trio (melody/bridge/piano), 1024 steps |
| `melody_octuple` | Event-based | 1 | Melody-only, variable length events |
| `trio_octuple` | Event-based | 3 | Piano trio, variable length events |

**Octuple format:** Each token is an 8-tuple:
```
[Bar, Position, Pitch, Duration, Velocity, Program, Tempo, Instrument]
```

### Available Tasks

Registered in `src/smdiff/tasks/registry.py`:

| Task ID | Description |
|---------|-------------|
| `uncond` | Unconditional generation (sample from scratch) |
| `infill` | Infilling/inpainting (fill masked regions given context) |

### Available Datasets

After running `prepare_data.py`, register in `src/smdiff/data/registry.py`:

| Dataset ID | Tokenizer | Bars | Notes | File |
|------------|-----------|------|-------|------|
| `pop909_melody` | `melody` | 64 | 1024 | `data/POP909_melody.npy` |
| `pop909_trio` | `trio` | 64 | 1024 | `data/POP909_trio.npy` |
| `pop909_melody_octuple` | `melody_octuple` | 64 | Variable | `data/POP909_melody_octuple.npy` |
| `pop909_trio_octuple` | `trio_octuple` | 64 | Variable | `data/POP909_trio_octuple.npy` |

---

## CLI Reference

### `prepare_data.py`

**Purpose:** Convert MIDI files to tokenized `.npy` caches.

**Minimal:**
```bash
python -m smdiff.cli.prepare_data \
    --root_dir data/POP909 \
    --tokenizer_id trio_octuple \
    --cache_path data/POP909_trio_octuple.npy
```

**Full options:**
```bash
python -m smdiff.cli.prepare_data \
    --root_dir data/POP909 \
    --tokenizer_id trio_octuple \
    --bars 64 \
    --cache_path data/POP909_trio_octuple.npy \
    --limit 0 \
    --num_workers 8 \
    --max_tensors_per_ns 5 \
    --strict_tempo
```

### `train.py`

**Purpose:** Train diffusion models on prepared datasets.

**Minimal:**
```bash
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple
```

**Full options:**
```bash
python train.py \
    --model octuple_ddpm \
    --dataset_id pop909_trio_octuple \
    --bars 64 \
    --batch_size 16 \
    --lr 1e-4 \
    --steps 100000 \
    --log_dir runs/my_experiment \
    --ema \
    --ema_rate 0.9999 \
    --resume
```

### `sample.py`

**Purpose:** Generate samples with trained models.

**Unconditional:**
```bash
python -m smdiff.cli.sample \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 16
```

**Infilling (single MIDI):**
```bash
python -m smdiff.cli.sample \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi data/POP909/001/001.mid \
    --mask_start_bar 16 \
    --mask_end_bar 32 \
    --n_samples 4
```

**Infilling (directory):**
```bash
python -m smdiff.cli.sample \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi_dir data/POP909 \
    --samples_per_midi 4 \
    --mask_start_bar 16 \
    --mask_end_bar 32
```

### `evaluate.py`

**Purpose:** Compute task-specific metrics on generated samples.

**Unconditional (generate):**
```bash
python -m smdiff.cli.evaluate \
    --task uncond \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --n_samples 100 \
    --save_samples
```

**Unconditional (load existing):**
```bash
python -m smdiff.cli.evaluate \
    --task uncond \
    --sample_dir runs/octuple_ddpm_trio_octuple/samples/uncond \
    --dataset_id pop909_trio_octuple
```

**Infilling (generate with multiple MIDIs):**
```bash
python -m smdiff.cli.evaluate \
    --task infill \
    --model octuple_ddpm \
    --load_dir runs/octuple_ddpm_trio_octuple \
    --dataset_id pop909_trio_octuple \
    --input_midi_dir data/POP909 \
    --samples_per_midi 4 \
    --n_samples 32 \
    --mask_start_bar 16 \
    --mask_end_bar 32
```

### `npz_to_midi.py`

**Purpose:** Convert `.npy` sample arrays (from training) to MIDI files.

**Single file:**
```bash
python npz_to_midi.py \
    --input runs/octuple_ddpm_trio_octuple/samples/sample_step_10000.npy \
    --tokenizer_id trio_octuple \
    --output_dir converted_samples/
```

**Batch conversion:**
```bash
for f in runs/*/samples/*.npy; do
    python npz_to_midi.py \
        --input "$f" \
        --tokenizer_id trio_octuple \
        --output_dir converted/
done
```

---

## Cluster Usage (SLURM)

Example batch scripts included:

**Training:**
```bash
sbatch run_training_job.sh
```

**Sampling:**
```bash
sbatch run_sampling_job.sh
```

**Evaluation:**
```bash
sbatch run_evaluation_job.sh
```

Edit scripts to set:
- `MODEL_ID`
- `RUN_DIR`
- `DATASET_ID`
- `N_SAMPLES`

---

## Attribution

This project builds upon:
- **SCHmUBERT**: Original symbolic music diffusion implementation
- **Unleashing Transformers**: Absorbing state diffusion framework
- **MusicBERT**: Octuple MIDI encoding inspiration

---

## License

See [LICENCE](LICENCE) for details.
