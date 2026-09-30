#!/usr/bin/env python3
"""Run the stage-18 GRPO trainer under instrumentation, without editing it.

Why this wrapper exists
-----------------------
`scripts/18_reinforcement-learning/05_train_grpo_human_writing.py` is kept
byte-for-byte as published so the baseline stays a faithful reproduction. But
as published it cannot support the measurements this project needs:

* its log and checkpoint directories are hardcoded module globals, so repeated
  runs append into one another's `metrics.csv` and `samples.txt`;
* `append_sample_logs` writes only the FIRST THREE rollouts of a group, which
  in practice hides the one rollout carrying the learning signal;
* it records no per-rollout record, no gradient norm, and no evidence about
  whether the optimizer actually moved any weights;
* it hardcodes a bfloat16 policy on CUDA.

That last point is not cosmetic. bfloat16 keeps 8 mantissa bits, so the spacing
between representable values near a typical Qwen3-0.6B weight (median |w| =
0.017) is about 1.2e-4. An AdamW step of size lr = 1e-5 is far below half that
spacing, so round-to-nearest leaves the weight unchanged: measured on the real
checkpoint, 87% of parameters cannot move at all. This is a property of the
bfloat16 FORMAT, not of any particular GPU -- native bf16 support buys speed,
not precision.

The same format costs reward resolution. The verifier softmaxes in bfloat16
before casting to float32, and the representable neighbours of 1.0 are
0.99609375 and 1.0. A confidently-detected rollout with true P(AI) = 0.9998
therefore becomes exactly 1.0, its human probability becomes exactly 0.0, and
its reward becomes exactly 0.0. Every confidently-detected rollout in a group
ties at zero, which destroys the within-group ORDERING that group-relative
advantages are computed from.

What this wrapper changes, and what it does not
-----------------------------------------------
It does NOT change the reward, the loss, the advantage computation, or any
hyperparameter default. The reward used for training is whatever the
unmodified trainer computes, in whatever dtype the policy runs. Sampling is
also unchanged unless `--uncap-response-tokens` is set. That flag replaces
`response_token_limit` with the raw `--max-new-tokens` ceiling so a short
target is not cut at `ceil(target_words * 1.6) + 16`. Off, the trainer's
cap is used exactly as published.

It DOES:
  * redirect the trainer's output globals to a per-run directory;
  * filter the prompt split to chosen target lengths;
  * choose the policy dtype explicitly (`--policy-dtype`);
  * record one JSONL row per rollout, including a float32 REFERENCE verifier
    score that is logged only and never touches a gradient;
  * record the pre-clipping gradient norm and a parameter-movement probe;
  * write a manifest pinning the commit, config, versions and prompt order.

Interception points are chosen so the training path is untouched:
`compute_grpo_loss` is wrapped only to carry the prompt example forward, and
`append_metrics` -- a pure logging sink -- is where instrumentation is written.
"""

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import subprocess
import sys
import time

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]
TRAINER_PATH = (
    PROJECT_DIR
    / "scripts"
    / "18_reinforcement-learning"
    / "05_train_grpo_human_writing.py"
)

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def load_trainer_module():
    """Import the unmodified stage-18 trainer as a module."""
    spec = importlib.util.spec_from_file_location("grpo_trainer", TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer: {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def git_state():
    def run(*command):
        try:
            return subprocess.run(
                command,
                cwd=PROJECT_DIR,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return None

    return {
        "commit": run("git", "rev-parse", "HEAD"),
        "branch": run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(run("git", "status", "--porcelain")),
    }


def package_versions():
    import importlib.metadata as metadata

    names = ("torch", "transformers", "datasets", "numpy", "scikit-learn")
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except Exception:
            versions[name] = None
    return versions


def load_policy_with_dtype(source, device, dtype, gradient_checkpointing):
    """Mirror of the trainer's `load_policy`, with the dtype made explicit.

    The trainer hardcodes `torch.bfloat16 if device.type == "cuda"`. Everything
    else here is identical to the original, deliberately.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    if tokenizer.eos_token_id is None:
        raise ValueError("Policy tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(source, dtype=dtype)
    model.to(device)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    return model, tokenizer


class MovementProbe:
    """Track whether optimizer steps actually change parameter values.

    A bfloat16 policy at lr = 1e-5 can run a whole training loop while most of
    its weights stay bit-identical. Loss and reward curves cannot show that, so
    it is measured directly on a fixed, deterministic sample of parameters.
    """

    def __init__(self, model, sample_size=200_000, seed=0):
        generator = torch.Generator().manual_seed(seed)
        self.entries = []
        remaining = sample_size
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad or parameter.numel() < 1_000:
                continue
            take = min(remaining, 20_000, parameter.numel())
            index = torch.randperm(parameter.numel(), generator=generator)[:take]
            self.entries.append((name, index))
            remaining -= take
            if remaining <= 0:
                break
        self.model = model
        self.initial = self._gather()
        self.previous = self.initial.clone()

    @torch.no_grad()
    def _gather(self):
        parameters = dict(self.model.named_parameters())
        chunks = [
            parameters[name].detach().flatten()[index.to(parameters[name].device)]
            .float()
            .cpu()
            for name, index in self.entries
        ]
        return torch.cat(chunks) if chunks else torch.zeros(0)

    @torch.no_grad()
    def measure(self):
        current = self._gather()
        since_start = (current - self.initial).abs()
        since_previous = (current - self.previous).abs()
        self.previous = current
        return {
            "probed_parameters": int(current.numel()),
            "moved_since_start_fraction": float((since_start > 0).float().mean()),
            "moved_this_step_fraction": float((since_previous > 0).float().mean()),
            "mean_abs_change_since_start": float(since_start.mean()),
            "max_abs_change_since_start": float(since_start.max())
            if current.numel()
            else 0.0,
        }


@torch.no_grad()
def reference_ai_probabilities(verifier, texts):
    """Score texts with the SAME verifier weights but a float32 softmax.

    Logged only. This never enters the reward, the advantages or the loss; it
    exists so the analysis can quantify how much ranking information the
    shipped bfloat16 path discarded.
    """
    safe = [text if text.strip() else "." for text in texts]
    scores = []
    for start in range(0, len(safe), 4):
        batch = safe[start : start + 4]
        encoded = {
            key: value.to(verifier.device)
            for key, value in verifier._prepare_batch(batch).items()
        }
        logits = verifier.model(**encoded).logits.float()
        probabilities = (logits / verifier.temperature).softmax(dim=1)
        scores.extend(probabilities[:, verifier.ai_index].cpu().tolist())
    return scores


def build_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset", default="rasbt/human-writing-prompts-6k")
    parser.add_argument("--policy-model", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--verifier-model", default="qwen3-variable")
    parser.add_argument(
        "--policy-dtype",
        choices=sorted(DTYPES),
        default="bfloat16",
        help=(
            "bfloat16 reproduces the published trainer exactly. float32 is the "
            "numerics-fixed variant: at lr=1e-5 a bfloat16 AdamW step rounds "
            "away for ~87%% of Qwen3-0.6B parameters."
        ),
    )
    parser.add_argument(
        "--target-words",
        type=int,
        nargs="+",
        default=[50, 100],
        help="Keep only prompts with these target lengths.",
    )
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--verifier-device", default="cuda")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "0-based index of the first training step. Step numbers are "
            "start-index+1 through --steps. Default 0 runs from step 1."
        ),
    )
    parser.add_argument("--num-rollouts", type=int, default=4)
    parser.add_argument("--rollout-batch-size", type=int, default=4)
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--uncap-response-tokens",
        action="store_true",
        help=(
            "Use --max-new-tokens as the generation cap. Off, the trainer "
            "still applies min(maximum, max(64, ceil(target_words * 1.6) + 16))."
        ),
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--log-samples-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-zero-advantage-updates", action="store_true")
    parser.add_argument(
        "--trigram-repetition",
        action="store_true",
        help=(
            "Multiply the reward by unique word-trigrams / all word-trigrams. "
            "Off, the reward stays P(human) * length_score."
        ),
    )
    parser.add_argument(
        "--kl-beta",
        type=float,
        default=0.0,
        help=(
            "Subtract beta * mean(log π_current - log π_base) over completion "
            "tokens. 0 leaves the reward at P(human) * length_score. The base "
            "is a frozen copy of --policy-model, eval mode, not optimized."
        ),
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--movement-probe-size",
        type=int,
        default=200_000,
        help="Number of parameter values tracked for the movement probe.",
    )
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="Permit writing into a non-empty run directory.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.kl_beta < 0:
        raise SystemExit("--kl-beta must be >= 0")

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.allow_existing:
        raise SystemExit(
            f"Run directory is not empty: {run_dir}\n"
            "Refusing to append into an existing run. Pick a new --run-dir or "
            "pass --allow-existing."
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    trainer = load_trainer_module()
    if args.uncap_response_tokens:
        trainer.response_token_limit = lambda target_words, maximum: maximum

    # Redirect the trainer's hardcoded output globals into this run's directory.
    trainer.LOG_DIR = run_dir
    trainer.CHECKPOINT_DIR = run_dir / "checkpoints"
    trainer.SAMPLE_LOG_PATH = run_dir / "samples.txt"
    trainer.METRICS_LOG_PATH = run_dir / "metrics.csv"

    # --- replicate the trainer's main() setup, in its original order ---------
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    policy_device = trainer.resolve_device(args.policy_device)

    from datasets import load_dataset

    train_data = load_dataset(args.dataset, split="train").shuffle(seed=args.seed)
    keep = set(args.target_words)
    train_data = train_data.filter(lambda row: int(row["target_words"]) in keep)
    if not train_data:
        raise ValueError("Filtered training split is empty")

    from ai_detector import load_classifier

    verifier = load_classifier(args.verifier_model, device=args.verifier_device)
    model, tokenizer = load_policy_with_dtype(
        args.policy_model,
        policy_device,
        DTYPES[args.policy_dtype],
        args.gradient_checkpointing,
    )
    # ------------------------------------------------------------------------

    probe = MovementProbe(model, sample_size=args.movement_probe_size)
    rollout_log = (run_dir / "rollouts.jsonl").open("a", encoding="utf-8")
    captured = {"example": None, "grad_norm": None}

    # Wrap compute_grpo_loss only to carry the prompt example into the sink.
    original_compute = trainer.compute_grpo_loss

    def compute_grpo_loss(*call_args, **call_kwargs):
        captured["example"] = (
            call_args[3] if len(call_args) > 3 else call_kwargs["example"]
        )
        captured["grad_norm"] = None
        return original_compute(*call_args, **call_kwargs)

    trainer.compute_grpo_loss = compute_grpo_loss

    # clip_grad_norm_ returns the norm BEFORE clipping; capture it in passing.
    original_clip = torch.nn.utils.clip_grad_norm_

    def clip_grad_norm_(*call_args, **call_kwargs):
        total = original_clip(*call_args, **call_kwargs)
        captured["grad_norm"] = float(total)
        return total

    torch.nn.utils.clip_grad_norm_ = clip_grad_norm_

    original_append_metrics = trainer.append_metrics

    def append_metrics(step, total_steps, stats, elapsed):
        original_append_metrics(step, total_steps, stats, elapsed)

        example = captured["example"] or {}
        target_words = int(example.get("target_words", 0) or 0)
        token_cap = (
            trainer.response_token_limit(target_words, args.max_new_tokens)
            if target_words > 0
            else None
        )
        samples = stats["samples"]
        reference = reference_ai_probabilities(
            verifier, [sample["text"] for sample in samples]
        )
        movement = probe.measure()

        record = {
            "step": step,
            "prompt_id": example.get("prompt_id"),
            "target_words": target_words or None,
            "question": example.get("prompt"),
            "token_cap": token_cap,
            "loss": stats["loss"],
            "optimizer_stepped": stats["loss_tensor"] is not None,
            "is_zero_advantage": bool(stats["is_zero_advantage"]),
            "grad_norm_preclip": captured["grad_norm"],
            "step_seconds": elapsed,
            "movement": movement,
            "rollouts": [
                {
                    "index": index,
                    "text": sample["text"],
                    "is_empty": not sample["text"].strip(),
                    "word_count": sample["word_count"],
                    "gen_tokens": sample["gen_len"],
                    "hit_token_cap": (
                        token_cap is not None and sample["gen_len"] >= token_cap
                    ),
                    # Training values, exactly as the trainer computed them.
                    "ai_probability": sample["ai_probability"],
                    "human_probability": sample["human_probability"],
                    "length_score": sample["length_score"],
                    **(
                        {"repetition_score": sample["repetition_score"]}
                        if "repetition_score" in sample
                        else {}
                    ),
                    **(
                        {"kl": sample["kl"]}
                        if "kl" in sample
                        else {}
                    ),
                    "reward": sample["reward"],
                    "advantage": stats["advantages"][index],
                    # Logged-only float32 reference; never touches a gradient.
                    "ai_probability_fp32": reference[index],
                    "human_probability_fp32": 1.0 - reference[index],
                    "reward_fp32": (1.0 - reference[index]) * sample["length_score"],
                }
                for index, sample in enumerate(samples)
            ],
        }
        rollout_log.write(json.dumps(record) + "\n")
        rollout_log.flush()

    trainer.append_metrics = append_metrics

    manifest = {
        "run_name": args.run_name or run_dir.name,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "argv": sys.argv,
        "config": {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "git": git_state(),
        "packages": package_versions(),
        "python": platform.python_version(),
        "trainer_path": str(TRAINER_PATH),
        "policy_dtype": str(next(model.parameters()).dtype),
        "verifier_dtype": str(next(verifier.model.parameters()).dtype),
        "verifier_temperature": verifier.temperature,
        "verifier_max_text_length": verifier.max_text_length,
        "prompts_after_filter": len(train_data),
        "prompt_ids_in_order": list(train_data["prompt_id"])[: args.steps],
    }
    if torch.cuda.is_available():
        manifest["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_native": bool(
                torch.cuda.is_bf16_supported(including_emulation=False)
            ),
        }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Run dir: {run_dir}")
    print(f"Policy dtype: {manifest['policy_dtype']}  device: {policy_device}")
    print(f"Verifier dtype: {manifest['verifier_dtype']}")
    print(f"Prompts after filter: {len(train_data)} (targets {args.target_words})")

    try:
        trainer.train(model, tokenizer, verifier, train_data, policy_device, args)
    finally:
        rollout_log.close()

    if torch.cuda.is_available():
        print(
            "Max CUDA memory allocated: "
            f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
        )


if __name__ == "__main__":
    main()
