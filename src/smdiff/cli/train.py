import sys
import os

# Ensure repository root is on sys.path so top-level packages like 'hparams' resolve
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Also ensure 'src' is on sys.path so 'smdiff' package resolves when running by path
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
    
import yaml
import argparse
from typing import Dict, List

from hparams.set_up_hparams import get_sampler_hparams
from smdiff.utils.log_utils import config_log, log, start_training_log, resolve_unique_log_dir
from smdiff import trainer

from smdiff.registry import resolve_model_id
from smdiff.configs.loader import load_config
from smdiff.data import apply_dataset_to_config
from smdiff.losses import resolve_loss_id
from smdiff.masking import resolve_masking_id
from smdiff.tokenizers import resolve_tokenizer_id
from smdiff.cluster import get_current_username, is_cluster, get_scratch_dir


def build_underlying_argv(cfg: Dict, ns: argparse.Namespace) -> List[str]:
    """Translate merged config + CLI into legacy hparams argv."""
    spec = resolve_model_id(ns.model)

    def pick(key, default=None):
        # CLI wins if explicitly provided; otherwise config
        val = getattr(ns, key, None)
        return val if val is not None else cfg.get(key, default)

    args = [
        "--model", spec.internal_model,
        "--dataset_path", pick("dataset_path"),
        "--batch_size", str(pick("batch_size")),
        "--lr", str(pick("lr")),
        "--bars", str(pick("bars")),
        "--tracks", pick("tracks"),
        "--steps_per_eval", str(pick("steps_per_eval")),
        "--steps_per_checkpoint", str(pick("steps_per_checkpoint")),
        "--steps_per_log", str(pick("steps_per_log")),
        "--steps_per_sample", str(pick("steps_per_sample")),
    ]

    epochs = pick("epochs")
    if epochs is not None:
        args += ["--epochs", str(epochs)]

    train_steps = pick("train_steps")
    if train_steps is not None:
        args += ["--train_steps", str(train_steps)]

    if pick("ema", True):
        args += ["--ema"]
    if pick("amp", False):
        args += ["--amp"]
    if pick("load_dir"):
        args += ["--load_dir", pick("load_dir")]
    if pick("load_step"):
        args += ["--load_step", str(pick("load_step"))]
    if pick("log_base_dir"):
        args += ["--log_base_dir", pick("log_base_dir")]
    if pick("port"):
        args += ["--port", str(pick("port"))]
    if pick("masking_strategy"):
        args += ["--masking_strategy", pick("masking_strategy")]
    if pick("loss_type"):
        args += ["--loss_type", pick("loss_type")]
    if pick("seed"):
        args += ["--seed", str(pick("seed"))]
    if pick("monotonicity_loss"):
        args += ["--monotonicity_loss"]

    loss_weights = pick("loss_weights")
    if loss_weights is not None:
        args += ["--loss_weights"] + [str(w) for w in loss_weights]

    lr_scheduler = pick("lr_scheduler")
    if lr_scheduler:
        args += ["--lr_scheduler", str(lr_scheduler)]
        
    warmup_iters = pick("warmup_iters")
    if warmup_iters is not None:
        args += ["--warmup_iters", str(warmup_iters)]

    eos = pick("eos")
    if eos is not None:
        args += ["--eos" if eos else "--no-eos"]

    return args


def main():
    parser = argparse.ArgumentParser(
        description="Unified training CLI for symbolic music diffusion/transformer models",
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Optional experiment config YAML to merge")
    parser.add_argument("--set", action="append", default=[],
                        help="Override config keys, e.g. --set lr=1e-4 --set batch_size=8")
    parser.add_argument("--model", required=True, type=str,
                        help="Model id: schmu_conv | schmu_tx | octuple_ddpm | octuple_mask_ddpm | musicbert_ddpm")
    parser.add_argument("--dataset_id", type=str, default=None,
                        help="Dataset id from DATASET_REGISTRY (e.g., pop909_melody, pop909_octuple)")
    parser.add_argument("--strategy", type=str, default=None,
                        help="Optional masking/training strategy id (passed as masking_strategy)")
    parser.add_argument("--loss_type", type=str, default=None,
                        help="Loss id from LOSS_REGISTRY (e.g., mmd_fmd_loss, mmd_loss, strict_fmd, plain_CE_loss, elbo, mlm)")

    # Common training settings (mapped to legacy parser)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument('--lr_scheduler', type=str, default=None, help='Learning rate scheduler (constant, cosine)')
    parser.add_argument('--warmup_iters', type=int, default=None, help='Number of warmup steps')
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--bars", type=int, default=None)
    parser.add_argument("--tracks", type=str, default=None)
    parser.add_argument("--monotonicity_loss", action="store_true", default=False)
    parser.add_argument("--loss_weights", nargs="+", type=float, default=None)
    parser.add_argument("--eos", action=argparse.BooleanOptionalAction, default=None,
                        help="Enable/disable EOS token insertion (--eos or --no-eos)")

    # Frequency/logging settings (kept compatible)
    parser.add_argument("--steps_per_eval", type=int, default=None)
    parser.add_argument("--steps_per_checkpoint", type=int, default=None)
    parser.add_argument("--steps_per_log", type=int, default=None)
    parser.add_argument("--steps_per_sample", type=int, default=None)
    parser.add_argument("--train_steps", type=int, default=None,
                        help="Override number of training steps (for quick smoke tests)")

    # Infra/quality-of-life
    parser.add_argument("--ema", action="store_true", default=None)
    parser.add_argument("--amp", action="store_true", default=None)
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--load_step", type=int, default=None)
    parser.add_argument("--log_base_dir", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--grad_acc", type=int, nargs='?', const=48, default=0, 
                        help="Target effective batch size. If used as a flag (--grad_acc), defaults to 48. Can also set specific value (--grad_acc 96).")
    
    # wanDB
    parser.add_argument("--wandb", const=True, action="store_const", default=False, help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="smdiff", help="WandB Project Name")
    parser.add_argument("--wandb_name", type=str, default=None, help="WandB Run Name")
    parser.add_argument("--local", action="store_true", help="Force local run (ignore cluster detection)")

    ns = parser.parse_args()

    # Ensure masking_strategy is available for pick() mechanism, mapping from --strategy
    if ns.strategy and not hasattr(ns, 'masking_strategy'):
        ns.masking_strategy = ns.strategy

    # Load and merge config
    cfg = load_config(ns.model, ns.config, ns.set)
    if ns.dataset_id:
        cfg = apply_dataset_to_config(cfg, ns.dataset_id)

    dataset_path = cfg.get("dataset_path")
    if not dataset_path or not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at '{dataset_path}'. Set --dataset_id or --dataset_path to an existing location."
        )

    tokenizer_id = cfg.get("tokenizer_id") or cfg.get("tracks", "melody")
    resolve_tokenizer_id(tokenizer_id)

    masking_strategy = cfg.get("masking_strategy") or ns.strategy
    if masking_strategy:
        resolve_masking_id(masking_strategy)

    selected_loss = ns.loss_type or cfg.get("loss_type") or "mmd_fmd_loss"
    loss_spec = resolve_loss_id(selected_loss)
    cfg["loss_type"] = loss_spec.id

    # Compose argv for existing hparams code
    translated_argv = [sys.argv[0]] + build_underlying_argv(cfg, ns)

    # Temporarily swap sys.argv to reuse legacy get_sampler_hparams
    prev_argv = sys.argv
    sys.argv = translated_argv
    try:
        H = get_sampler_hparams('train')
    finally:
        sys.argv = prev_argv
        
    run_id = f"{ns.model}_{tokenizer_id}"
    if masking_strategy:
        run_id += f"_{masking_strategy}"
        
    project_run_dir = os.path.join("runs", run_id)
    H.project_log_dir = os.path.abspath(project_run_dir)
    
    # 2. Define the Active Log Path (Scratch vs Home)
    if is_cluster() and not ns.local:
        username = get_current_username()
        scratch_root = get_scratch_dir(username)
        # /work/scratch/user/runs/model_id
        H.log_dir = os.path.join(scratch_root, "runs", run_id)
        print(f"Cluster detected: Logging to Scratch ({H.log_dir})")
    else:
        H.log_dir = H.project_log_dir
        print(f"Local run: Logging to Project Dir ({H.log_dir})")

    # Enrich hparams for logging/visibility
    H.tokenizer_id = tokenizer_id
    H.dataset_id = ns.dataset_id
    H.model_id = ns.model  # Store canonical model_id for registry lookup
    H.grad_acc = ns.grad_acc
    H.loss_type = cfg.get("loss_type", "mmd_fmd_loss")
    H.mmd_fmd = cfg.get("mmd_fmd", {})
    H.strict_fmd = cfg.get("strict_fmd", {})
    H.sync_bar_ddpm = cfg.get("sync_bar_ddpm", {})
    H.sync_bar_mdlm = cfg.get("sync_bar_mdlm", {})
    H.octuple_mdlm = cfg.get("octuple_mdlm", {})
    H.hierarchical_masking = cfg.get("hierarchical_masking", {})

    H.wandb = ns.wandb
    H.wandb_name = ns.wandb_name
    H.wandb_project = ns.wandb_project

    # Avoid overwriting existing run folders by suffixing an index when needed.
    resolved_log_dir, run_suffix = resolve_unique_log_dir(H.log_dir)
    if run_suffix is not None:
        H.log_dir = resolved_log_dir
        # Keep scratch->home sync target unique as well.
        project_candidate = f"{H.project_log_dir}_{run_suffix}"
        H.project_log_dir, _ = resolve_unique_log_dir(project_candidate)
        print(f"Run directory exists, using indexed path ({H.log_dir})")
        print(f"Project sync target set to ({H.project_log_dir})")

    if not H.load_dir:
        H.load_dir = H.project_log_dir

    # Proceed with training
    config_log(H.log_dir)
    # Snapshot effective config and CLI into runs/{model}/configs
    try:
        cfg_dir = os.path.join(H.log_dir, "configs")
        os.makedirs(cfg_dir, exist_ok=True)
        # Save merged config
        with open(os.path.join(cfg_dir, "config.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=True)
        # Save resolved hparams (lightweight)
        with open(os.path.join(cfg_dir, "hparams.yaml"), "w", encoding="utf-8") as f:
            yaml.safe_dump(dict(H), f, sort_keys=True)
        # Save command line
        with open(os.path.join(cfg_dir, "command.txt"), "w", encoding="utf-8") as f:
            f.write(" ".join(sys.argv))
    except Exception as e:
        print(f"Warning: failed to snapshot config: {e}")
    log('---------------------------------')
    log(f'Setting up training for {H.sampler} (model={H.model})')
    start_training_log(H)
    trainer.main(H)


if __name__ == "__main__":
    main()
