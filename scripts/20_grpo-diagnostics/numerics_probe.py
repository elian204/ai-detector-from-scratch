#!/usr/bin/env python3
"""Measure the numerical headroom of the stage-18 GRPO setup on this machine.

Stage 18 trains a bf16 policy (05_train_grpo_human_writing.py hardcodes
bfloat16 on CUDA) and rewards it with a bf16 verifier whose softmax is taken
*before* the cast to float32 (src/ai_detector/classifiers.py). On Turing GPUs
(Quadro RTX 6000, compute capability 7.5) bf16 is emulated, not native.

Two consequences are measured here, because both can make a GRPO run look
like "no reward hacking observed" for purely numerical reasons:

  A. Update resolution. A bf16 AdamW step of size lr can be smaller than half
     the bf16 spacing (ULP) at the parameter's magnitude, so it rounds away and
     the weight never changes.

  B. Reward resolution. bf16 has ~3 decimal digits near 1.0: the representable
     neighbours of 1.0 are 0.99609375 and 1.0. A verifier that is confident
     enough (P_AI >= ~0.998) returns exactly 1.0, so human_probability = 1 - P_AI
     is exactly 0.0 and the reward is exactly 0.0 no matter how the text differs.
     A whole group of such rollouts has zero reward variance, hence zero
     advantages and no learning signal.

Nothing here modifies the training script or the reward. This is measurement
only; it is meant to be run before the baseline so the baseline's outcome can
be attributed to the algorithm rather than to float formats.
"""

import argparse
import json
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parents[1]

from ai_detector import load_classifier, model_registry  # noqa: E402


def bf16_ulp(magnitudes):
    """Spacing between adjacent bf16 values at the given magnitudes.

    bf16 has 8 explicit mantissa bits (7 stored), so the spacing at a value in
    [2^e, 2^(e+1)) is 2^(e-7).
    """
    safe = magnitudes.clamp(min=1e-30)
    return torch.pow(2.0, torch.floor(torch.log2(safe)) - 7)


def probe_update_resolution(weights_path, learning_rate, sample_per_tensor=20_000):
    """Fraction of real policy-shaped weights a single lr-sized step cannot move.

    A round-to-nearest update of size ``lr`` leaves a bf16 weight unchanged when
    ``lr < ulp/2``, i.e. when ``ulp > 2*lr``.
    """
    from safetensors import safe_open

    sampled = []
    with safe_open(str(weights_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            if not key.endswith(".weight"):
                continue
            tensor = handle.get_tensor(key)
            if tensor.dtype != torch.bfloat16 or tensor.numel() < 1_000:
                continue
            flat = tensor.abs().flatten().float()
            take = min(sample_per_tensor, flat.numel())
            index = torch.randperm(flat.numel())[:take]
            sampled.append(flat[index])
    if not sampled:
        raise ValueError(f"No bf16 weight tensors found in {weights_path}")
    magnitudes = torch.cat(sampled)
    frozen = bf16_ulp(magnitudes) > 2.0 * learning_rate
    return {
        "sampled_parameters": int(magnitudes.numel()),
        "median_abs_weight": float(magnitudes.median()),
        "mean_abs_weight": float(magnitudes.mean()),
        "learning_rate": learning_rate,
        "frozen_fraction": float(frozen.float().mean()),
    }


def probe_adamw_empirically(learning_rate, steps=20, size=4_096):
    """Run real AdamW on bf16 and fp32 copies of the same weights."""
    results = {}
    for dtype in (torch.bfloat16, torch.float32):
        torch.manual_seed(0)
        start = torch.empty(size).normal_(0.0, 0.02).to(dtype)
        weight = torch.nn.Parameter(start.clone())
        optimizer = torch.optim.AdamW([weight], lr=learning_rate)
        before = weight.detach().float().clone()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            # Constant unit gradient: the cleanest possible signal. AdamW
            # normalizes it, so each step has magnitude ~= lr.
            weight.backward(torch.ones_like(weight))
            optimizer.step()
        delta = (weight.detach().float() - before).abs()
        results[str(dtype).replace("torch.", "")] = {
            "steps": steps,
            "moved_fraction": float((delta > 0).float().mean()),
            "mean_abs_delta": float(delta.mean()),
        }
    return results


def reward_from_ai_probability(ai_probability, word_count, target_words):
    """The stage-18 reward, reimplemented here so the probe stays standalone."""
    if word_count <= 0:
        return 0.0
    length_score = min(word_count, target_words) / max(word_count, target_words)
    return (1.0 - ai_probability) * length_score


def probe_reward_resolution(texts, *, device, batch_size):
    """Score the same texts with the shipped bf16 verifier and an fp32 copy.

    The fp32 pass is the reference: it keeps the softmax in fp32, so it shows
    how much probability resolution the bf16 path discards.
    """
    verifier = load_classifier("qwen3-variable", device=device)
    bf16_scores = verifier.score_many(texts, batch_size=batch_size)

    # Same weights, same temperature, but softmax in fp32.
    model = verifier.model
    fp32_scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = {
            key: value.to(verifier.device)
            for key, value in verifier._prepare_batch(batch).items()
        }
        with torch.inference_mode():
            logits = model(**encoded).logits.float()
            probabilities = (logits / verifier.temperature).softmax(dim=1)
            fp32_scores.extend(
                probabilities[:, verifier.ai_index].cpu().tolist()
            )

    bf16_tensor = torch.tensor(bf16_scores, dtype=torch.float64)
    fp32_tensor = torch.tensor(fp32_scores, dtype=torch.float64)
    summary = {
        "texts_scored": len(texts),
        "model_dtype": str(next(model.parameters()).dtype),
        "temperature": verifier.temperature,
        "bf16": {
            "exactly_one_fraction": float((bf16_tensor == 1.0).float().mean()),
            "distinct_values": int(torch.unique(bf16_tensor).numel()),
            "min": float(bf16_tensor.min()),
            "max": float(bf16_tensor.max()),
            "mean": float(bf16_tensor.mean()),
        },
        "fp32": {
            "exactly_one_fraction": float((fp32_tensor == 1.0).float().mean()),
            "distinct_values": int(torch.unique(fp32_tensor).numel()),
            "min": float(fp32_tensor.min()),
            "max": float(fp32_tensor.max()),
            "mean": float(fp32_tensor.mean()),
        },
        "max_abs_difference": float((bf16_tensor - fp32_tensor).abs().max()),
        "bf16_human_prob_exactly_zero_fraction": float(
            ((1.0 - bf16_tensor) == 0.0).float().mean()
        ),
        "fp32_human_prob_exactly_zero_fraction": float(
            ((1.0 - fp32_tensor) == 0.0).float().mean()
        ),
    }
    return summary, bf16_scores, fp32_scores


def group_zero_advantage_rate(ai_probabilities, word_counts, targets, group_size):
    """How often a group of rollouts would yield exactly zero advantages.

    GRPO divides by the group's reward standard deviation; when every rollout in
    a group scores identically the advantages are exactly zero and the step
    carries no signal.
    """
    rewards = [
        reward_from_ai_probability(p, w, t)
        for p, w, t in zip(ai_probabilities, word_counts, targets, strict=True)
    ]
    groups = [
        rewards[start : start + group_size]
        for start in range(0, len(rewards) - group_size + 1, group_size)
    ]
    zero_groups = sum(
        1 for group in groups if len(set(group)) == 1
    )
    all_zero_reward = sum(1 for group in groups if all(r == 0.0 for r in group))
    return {
        "group_size": group_size,
        "groups": len(groups),
        "zero_advantage_fraction": zero_groups / len(groups) if groups else None,
        "all_zero_reward_fraction": all_zero_reward / len(groups) if groups else None,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__,
        epilog=(
            "Without --texts the reward probes are skipped and only the "
            "parameter-update probes run."
        ),
    )
    parser.add_argument(
        "--texts",
        type=Path,
        default=None,
        help=(
            "JSONL with one object per line containing a 'text' field and "
            "optionally 'target_words'. Typically base-policy completions."
        ),
    )
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--target-words", type=int, default=100)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument(
        "--device", choices=("auto", "cuda", "mps", "cpu"), default="auto"
    )
    parser.add_argument("--verifier-batch-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    report = {"learning_rate": args.learning_rate}

    if torch.cuda.is_available():
        report["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "bf16_supported_natively": bool(
                torch.cuda.is_bf16_supported(including_emulation=False)
            ),
        }

    report["adamw_empirical"] = probe_adamw_empirically(args.learning_rate)

    spec = model_registry()["qwen3-variable"]
    weights = spec.artifact_path / "model.safetensors"
    if weights.is_file():
        report["update_resolution"] = probe_update_resolution(
            weights, args.learning_rate
        )
    else:
        report["update_resolution"] = f"skipped: {weights} is missing"

    if args.texts is not None:
        rows = [
            json.loads(line)
            for line in args.texts.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [row for row in rows if str(row.get("text", "")).strip()]
        if not rows:
            raise ValueError(f"No non-empty 'text' values in {args.texts}")
        texts = [row["text"] for row in rows]
        resolution, bf16_scores, fp32_scores = probe_reward_resolution(
            texts, device=args.device, batch_size=args.verifier_batch_size
        )
        report["reward_resolution"] = resolution

        word_counts = [len(text.split()) for text in texts]
        targets = [int(row.get("target_words", args.target_words)) for row in rows]
        report["zero_advantage"] = {
            "note": (
                "groups are consecutive runs of --group-size texts in file "
                "order; a group with identical rewards yields exactly zero "
                "advantages and therefore no gradient signal"
            ),
            "bf16": group_zero_advantage_rate(
                bf16_scores, word_counts, targets, args.group_size
            ),
            "fp32": group_zero_advantage_rate(
                fp32_scores, word_counts, targets, args.group_size
            ),
        }

    text = json.dumps(report, indent=2)
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
