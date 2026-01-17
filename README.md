# Symbolic Music Discrete Diffusion

A symbolic music generation framework based on absorbing state diffusion, supporting both grid-based and Octuple MIDI representations.

This implementation provides flexible tools for training, and evaluating discrete diffusion models on symbolic music data, with support for unconditional generation and infilling tasks.

The goal is to introduce explicit Octuple encoding for MIDI data and compare different masking strategies enabled by the representation with current grid-based implementations. To this end, the SchmuBERT repository was forked to build on the exisitng work done for grid-based encoders.

---

## Dataset: POP909

The dataset is the highly curated [POP909 dataset](https://github.com/music-x-lab/POP909-Dataset). it natively has three tracks: MELODY, PIANO, BRIDGE. All tracks are played by the same MIDI intrument however: Grand Acoustic Piano.

The original SchmuBERT implementation does not support this track distinction, so we enhanced the conversion pipelines to keep the structure of the POP909 songs. POP909 songs come with tempo annotation and tempo changes natively, but to be able ro process them fully when converting to wncoded format, we normalized a song by choosing the starting tempo as the general tempo for the whole song. Given these are all pop songs, this is a valid simplification to make.

## Available Components

### Available Models

Registered in `src/smdiff/registry/models.py`:

| Model ID | Architecture | Description |
|----------|-------------|-------------|
| `octuple_ddpm` | Absorbing Diffusion | DDPM for octuple encoding |
| `octuple_mask_ddpm` | Absorbing Diffusion with masking strategies | DDPM for octuple  encoding |
| `musicbert_ddpm`* | MusicBERT Transformer-Encoder + Absorbing Diffusion (Transformer) | DDPM for octuple  encoding |
| `schmu_tx_vae` | Transformer VAE + Absorbing Diffusion | Grid-based transformer VAE |
| `schmu_conv_vae` | Convolutional VAE + Absorbing Diffusion | Grid-based conv VAE |

\* The MusicBERT transformer-encoder must be pretrained on the same dataset. Use `scripts\run_musicBERT_pre_training.sh`. It does not support logging to wanDB.

### Available Tokenizers

Registered in `src/smdiff/tokenizers/registry.py`:

| Tokenizer ID | Type | Channels | Description |
| ------------ | ---- | -------- | ----------- |
| `melody` | Grid | 1 | Melody-only, 1024 token (64 bars × 16 token) |
| `trio` | Grid | 3 | Piano trio (melody/bridge/piano), 1024 token |
| `melody_octuple` | Event-based | 1 | Melody-only, variable length events (padded to 1024 tokens*) |
| `trio_octuple` | Event-based | 8 | Piano trio, variable length events (padded to 1024 tokens*) |

\* `Padding Token = -1` in each subtoken of the Octuple encoding (a `-1` anywhere in the octuple thus marks an invalid token).

#### Octuple format

Since POP909 songs are played only by MIDI instrument _Grand Acoustic Piano_, we modified the `Instrument` subtoken to represent the _Trio_ tracks of the POP909 songs instead. By default the token hence takes values `[0, 1, 2]`.

```python
[Bar, Position, Instrument, Pitch, Duration, Velocity, TimeSignature, Tempo]
```

### Available Tasks

Registered in `src/smdiff/tasks/registry.py`:

| Task ID | Description |
| --------- | ------------- |
| `uncond` | Unconditional generation (sample from scratch) |
| `infill` | Infilling/inpainting (fill masked regions given context) |

### Available Strategies

Registered in `src/smdiff/masking/registry.py`:

| Strategy | Description |
| -------- | ----------- |
| `random` | Token-level masking: mask whole token (all channels) at random positions. |
| `bar_all` | Dynamic bar-level masking: masks K bars where K is proportional to t/T. Implementation masks attributes {pitch,duration,velocity,tempo}. K scales linearly with timestep. |
| `bar_attribute` | Dynamic attribute-level masking: masks K (bar, attribute) pairs where K is proportional to t/T. Attribute is chosen from {pitch,duration,velocity,tempo}. K scales linearly with timestep. |
| `mixed` | Randomly choose one masking strategy per batch (includes 'random'). |
| `sync_bar` | Dynamic attribute-level masking: masks K (bar, attribute) pairs where K is proportional to t/T, extend logic to bars. Attribute is chosen from {pitch,duration,velocity,tempo}. K scales linearly with timestep. |
| `sync_bar_position` | Dynamic attribute-level masking: masks K (bar, attribute) pairs where K is proportional to t/T, extend logic to bars & position. Attribute is chosen from {pitch,duration,velocity,tempo}. K scales linearly with timestep. |

## Codebase

All shell scripts for starting `sbatch` jobs on the DINFK cluster are in the `scripts` folder.
All entry points for training, evaluation and converting files are registered as python modules in `src\smdiff\cli`.
Outputs are saved to the `runs` folder (see below).

The `tests` folder contains some inspection tools for generated sampels (npy format) or datasets. Output plots are saved to the `plots` folder.

### Training & Evaluation Output

All training output will be available under the `runs` directory. When running on the DINFK cluster, the full output and intermediate model weights can be obtained from the `scratch` directory. Only the `best` and `latest` model are copied to the project directory. The evaluation script will also use the `scratch` directory as a fallback to look for weights if none are present in the project directory `runs` folder.

```bash
runs/<model>_<tokenizer_id>_<masking_strategy>/
├── checkpoints/
│   ├── best.pt              # Best validation checkpoint
│   ├── ema_best.pt          # Best EMA checkpoint
│   ├── model_<step>.th      # Regular checkpoints
│   └── ema_<step>.th
├── configs/
│   └── config.yaml          # Merged config used for training
├── samples/                 # Generated during training
│   └── *.npy
├── metrics/                 # Generated during evaluation
│   └── *.json
├── samples/                 # Generated during training
│   └── *.npy
├── stats/                   # Generated during training
│   └── *.pt
└── logs/                    # Training logs
```

## Environment

### Setup

```bash
# Clone repository
git clone https://github.com/Nengeldir/Octubert.git
cd Octubert/

python -m venv .venv
source .venv/bin/activate       # Linux/Mac
.venv\Scripts\activate          # Windows
pip install -r requirements.txt # On the DINFK cluster, use requirements_cluster.txt (includes nvidia binaries)
```

Make sure to enable the cuda 12.8 module when running on the DINFK cluster.
See the cluster [documentation](<https://www.isg.inf.ethz.ch/Main/HelpClusterComputingStudentCluster>) for details.

#### wanDB

The training process is configured to log to wanDB if the cli flags are set. To set up your machine, activate your python environment, run below command and paste your API key from wanDB.

```bash
wandb login
```

### Preparing the data

To prepare the data, you must run `prepare_data.py` for every tokenizer you'd like to train on later.
The `prepare_data.py` script converts MIDI files into tokenized `.npy` caches for training. YOu will find the results in `data` folder.

```bash
python -m smdiff.cli.prepare_data --tokenizer_id trio_octuple
```

### Training the models

To train the models, you must run `train.py` with dataset and model specification. All other configuration parameters overwrite the default settings. In case you'd like to run the `octuple_mask_ddpm` model, you must provide `--strategy bar_all|bar_attribute|mixed|sync_bar|sync_bar_position` as well. Exclude the wandb flags if you do not want to track progress online. Please see the example below:

```bash
python src/smdiff/cli/train.py \
  --model octuple_ddpm \
  --dataset_id pop909_trio_octuple \
  --batch_size 4 \
  --epochs 100 \
  --steps_per_log 10 \
  --steps_per_eval 1000 \
  --steps_per_sample 5000 \
  --steps_per_checkpoint 5000 \
  --seed 67 \
  --wandb \
  --wandb_project "octubert-music" \
  --wandb_name "octuple-ddpm-trio-octuple"
```

To inspect training progress, next to train/val loss logging, the model generates samples in `npy` format every `steps_per_sample`. To convert these to `mid` files, run `npy_to_midi.py` and specify the run directory. The output will be saved in a `midi` subfolder within the `runs/<model>_<tokenizer>_<strategy>/samples` folder.

```bash
python -m src.smdiff.cli.npy_to_midi --run_dir runs/schmu_conv_vae_trio
```

### Evaluating the models

For evaluation, you must specify a `task` to evaluate: Either `uncond` or `infill`.

- `uncond`: Evaluation will per default generate 100 unconditional samples and then run the metrics suite to evaluate the generated music against the original dataset (based on tokenizer used).
- `infill`: Evaluation will per default grab all `*.mid` files from `data\test\POP909` and extract sequences of `1024` token, mask region `256-512` and generate `2` conditional samples per sequence. The evaluation suite then aims to quanitfy the reconstruction success of the model against the orignal test data.

Note that this is a purely mathematical and statistical analysis.

Call `evaluate_octuple.py` to evaluate octuple-encoded data and `evaluate_trio.py`for trio-encoded data. If no `load_step` is supplied, the evaluation automatically uses `best` weights. Note that `best` weights might have occured in an early training step due to fluctuation in validation loss.

Examples: 

```bash
python -m smdiff.cli.evaluate_octuple \
  --task uncond \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --n_samples 100 \
  --batch_size 4 \
  --load_step 63000
```

```bash
python3 -m smdiff.cli.evaluate_octuple \
  --task infill \
  --model octuple_ddpm \
  --load_dir runs/octuple_ddpm_trio_octuple \
  --input_midi_dir data/test/POP909 \
  --batch_size 4 \
  --mask_token_start 256 \
  --mask_token_end 512 \
  --load_step 63000
```

```bash
python3 -m smdiff.cli.evaluate_trio \
  --task uncond \
  --model "${MODEL_ID}" \
  --load_dir "${RUN_DIR}" \
  --batch_size 4 \
  --tracks trio                   # Note tracks argument must be supplied additionally when running evaluate_trio.py
```

#### Output

A complete description of the metrics computed can be found the [Metrics README](METRICS.md).
Metrics are saved to `runs/<model>/metrics/metrics_<task>_<load_step|best>.json`:

```json
{
  "pch_kl": 0.03601442965017303,
  "duration_kl": 0.3711027813647103,
  "velocity_kl": 0.3666322734044071,
  "note_density_kl": 1.5803544345426206,
  "self_similarity": 0.7061530456502605,
  "self_similarity_std": 0.28858907552327906,
  "pitch_range_mean": 75.56,
  "pitch_range_std": 10.262865097037961,
  "sample_diversity": 380.4882856629088,
  "valid_samples_pct": 100.0
}
```

## Listen to MIDI files

Use an extension in VS Code. Should this not work out of the box, try placing a `soundfont.sf2` file in the project directory additionally.

## Attribution

This project builds upon:

- **SchmuBERT**: Original symbolic music diffusion implementation
- **Magenta VAE pipelines**: Pipelines for encoding MIDI data in grid-based format
- **MusicBERT**: Octuple MIDI encoding inspiration

## License

See [LICENCE](LICENCE) for details.
