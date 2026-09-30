
import importlib.util
from pathlib import Path
import sys

import pytest
import torch


PROJECT_DIR = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_DIR
    / "scripts"
    / "18_reinforcement-learning"
    / "05_train_grpo_human_writing.py"
)
SPEC = importlib.util.spec_from_file_location("grpo_human_writing", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeVerifier:
    def score_many(self, texts, *, batch_size):
        assert batch_size == 2
        assert texts == ["one two three four", "one two"]
        return [0.25, 0.75]


def test_render_prompt_carries_question_and_target_length():
    prompt = MODULE.render_prompt(
        {"prompt": "How does attention work?", "target_words": 250}
    )

    assert "approximately 250 words" in prompt
    assert "Question: How does attention work?" in prompt
    assert prompt.endswith("Answer:")


@pytest.mark.parametrize(
    ("word_count", "target_words", "expected"),
    [(0, 100, 0.0), (50, 100, 0.5), (100, 100, 1.0), (200, 100, 0.5)],
)
def test_length_adherence_penalizes_short_and_long_answers(
    word_count, target_words, expected
):
    assert MODULE.length_adherence_score(word_count, target_words) == expected


def test_trigram_repetition_score_penalizes_a_loop():
    loop = " ".join(["of", "the", "U.S."] * 20)
    prose = (
        "Legal history is shaped by political social economic and cultural "
        "influences today"
    )

    assert MODULE.trigram_repetition_score(loop) < 0.2
    assert MODULE.trigram_repetition_score(prose) > 0.9
    assert MODULE.trigram_repetition_score("one two") == 1.0


class ConstantVerifier:
    def score_many(self, texts, *, batch_size):
        assert batch_size == 1
        return [0.2] * len(texts)


def test_flag_off_reward_ignores_trigram_repetition():
    loop = " ".join(["of", "the", "U.S."] * 8)
    off_rewards, off_details = MODULE.compute_human_writing_rewards(
        [loop],
        target_words=24,
        verifier=ConstantVerifier(),
        verifier_batch_size=1,
    )
    on_rewards, on_details = MODULE.compute_human_writing_rewards(
        [loop],
        target_words=24,
        verifier=ConstantVerifier(),
        verifier_batch_size=1,
        trigram_repetition=True,
    )

    length_score = MODULE.length_adherence_score(len(loop.split()), 24)
    repetition_score = MODULE.trigram_repetition_score(loop)
    assert off_rewards == pytest.approx([0.8 * length_score])
    assert "repetition_score" not in off_details[0]
    assert repetition_score < 0.2
    assert on_rewards == pytest.approx([0.8 * length_score * repetition_score])
    assert on_details[0]["repetition_score"] == pytest.approx(repetition_score)


def test_reward_requires_both_human_score_and_length_adherence():
    rewards, details = MODULE.compute_human_writing_rewards(
        ["one two three four", "one two"],
        target_words=4,
        verifier=FakeVerifier(),
        verifier_batch_size=2,
    )

    assert rewards == pytest.approx([0.75, 0.125])
    assert details[0]["human_probability"] == pytest.approx(0.75)
    assert details[1]["length_score"] == pytest.approx(0.5)


def test_group_advantages_are_centered_and_finite():
    advantages = MODULE.normalized_advantages(
        [0.1, 0.2, 0.3, 0.4], torch.device("cpu")
    )

    assert torch.isfinite(advantages).all()
    assert float(advantages.mean()) == pytest.approx(0.0, abs=1e-6)


def test_rollout_token_cap_scales_with_requested_word_count():
    assert MODULE.response_token_limit(50, 1616) == 96
    assert MODULE.response_token_limit(1000, 1616) == 1616
    assert MODULE.response_token_limit(1000, 512) == 512
