from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import pytest

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import (
    MDNAuxiliaryTrainer,
    MDNAuxiliaryTrainerConfig,
    build_auxiliary_record,
)
from generator.train_mdn_auxiliary import train_auxiliary_from_records


def _baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": (0.5, 0.2),
    }


def _synthetic_records() -> list:
    records = []
    for index in range(40):
        context = (0.1,) * 14 if index % 2 == 0 else (0.9,) * 14
        skill_id = 1 if index % 2 == 0 else 2
        payoff = 1.7 if index % 2 == 0 else 1.1
        motives = (0.8, 0.4) if index % 2 == 0 else (0.3, 0.7)
        records.append(
            build_auxiliary_record(
                context=context,
                skill_id=skill_id,
                payoff=payoff,
                motives=motives,
                baseline_stats=_baseline_stats(),
            )
        )
    return records


def test_auxiliary_trainer_runs_and_saves_checkpoint(tmp_path: Path):
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            checkpoint_path=str(tmp_path / "mdn_auxiliary_best.pth"),
            max_epochs=10,
            patience=3,
            batch_size=8,
            random_seed=0,
        ),
        device="cpu",
    )

    result = trainer.train_records(_synthetic_records())

    assert Path(result["checkpoint_path"]).exists()
    assert result["best_val_loss"] >= 0.0
    assert result["best_metrics"]["epoch"] >= 1


def test_auxiliary_trainer_improves_gate_accuracy_and_q_error():
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(
            max_epochs=15,
            patience=5,
            batch_size=8,
            random_seed=0,
        ),
        device="cpu",
    )

    result = trainer.train_records(_synthetic_records())
    train_metrics = result["best_metrics"]["train"]

    assert 0.0 <= train_metrics["gate_accuracy"] <= 1.0
    assert torch.isfinite(torch.tensor(train_metrics["loss"]))
    assert torch.isfinite(torch.tensor(train_metrics["q_loss"]))


def test_auxiliary_trainer_supports_huber_q_loss():
    model = MotiveDecompositionNetwork(input_dim=2, num_skills=4, num_objectives=1)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(q_loss="huber", huber_delta=1.0),
        device="cpu",
    )

    total_loss, gate_loss, q_loss = trainer._compute_losses(
        gate_logits=torch.tensor([0.0]),
        q_hat=torch.tensor([[10.0]]),
        accept_label=torch.tensor([1.0]),
        q_target=torch.tensor([[0.0]]),
    )

    assert q_loss.item() == pytest.approx(9.5)
    assert total_loss.item() == pytest.approx(gate_loss.item() + 0.1 * 9.5)


def test_build_auxiliary_record_uses_discounted_target_when_trajectory_exists():
    record = build_auxiliary_record(
        context=(0.1,) * 14,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[(1.0, 0.0), (0.0, 1.0)],
        gamma=0.5,
    )

    assert len(record.q_target) == 2


def test_build_auxiliary_record_raises_for_ips_without_probabilities():
    try:
        build_auxiliary_record(
            context=(0.1,) * 14,
            skill_id=1,
            payoff=1.2,
            motives=(0.5, 0.2),
            baseline_stats=_baseline_stats(),
            motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
            use_ips=True,
        )
    except ValueError as exc:
        assert "behavior_probability" in str(exc)
    else:
        raise AssertionError("Expected ValueError when IPS probabilities are missing")


def test_build_auxiliary_record_preserves_behavior_probability_when_present():
    record = build_auxiliary_record(
        context=(0.1,) * 8,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
        behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
        target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        record_behavior_probability=0.5,
        use_ips=True,
        all_candidate_delta_r=(0.4, 0.2),
        all_candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
        selected_candidate_index=0,
    )

    assert record.behavior_probability == 0.5
    assert record.candidate_delta_r == (0.4, 0.2)
    assert record.selected_candidate_index == 0


def test_build_auxiliary_record_keeps_discounted_target_for_probability_aware_ips():
    record = build_auxiliary_record(
        context=(0.1,) * 8,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
        behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
        target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        record_behavior_probability=0.5,
        use_ips=True,
        all_candidate_delta_r=(0.4, 0.2),
        all_candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
        selected_candidate_index=0,
    )

    assert np.allclose(record.q_target, (1.0, 1.0))


def test_auxiliary_trainer_raises_when_ips_mode_enabled_without_probability_aware_dataset():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, batch_size=8, max_epochs=1),
        device="cpu",
    )

    try:
        trainer.train_records(_synthetic_records())
    except ValueError as exc:
        assert "train_probability_aware_records" in str(exc)
    else:
        raise AssertionError("Expected ValueError when IPS mode is enabled without probability-aware dataset support")


def test_auxiliary_trainer_raises_when_dr_mode_enabled_without_probability_aware_dataset():
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=16, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, batch_size=8, max_epochs=1),
        device="cpu",
    )

    with pytest.raises(ValueError, match="train_probability_aware_records"):
        trainer.train_records(_synthetic_records())


def test_probability_aware_auxiliary_training_path_runs(tmp_path: Path):
    records = []
    for index in range(20):
        records.append(
            build_auxiliary_record(
                context=((0.1,) * 8) if index % 2 == 0 else ((0.9,) * 8),
                skill_id=1 if index % 2 == 0 else 2,
                payoff=1.7 if index % 2 == 0 else 1.1,
                motives=(0.8, 0.4) if index % 2 == 0 else (0.3, 0.7),
                baseline_stats=_baseline_stats(),
                motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
                behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
                target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
                record_behavior_probability=0.5,
                use_ips=True,
                all_candidate_delta_r=(0.4, 0.2),
                all_candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
                selected_candidate_index=0,
            )
        )

    result = train_auxiliary_from_records(
        records,
        checkpoint_path=str(tmp_path / "mdn_auxiliary_ips_best.pth"),
        seed=0,
        device="cpu",
        use_ips=True,
    )

    assert Path(result["checkpoint_path"]).exists()
    assert result["best_val_loss"] >= 0.0


def test_train_auxiliary_from_records_exposes_doubly_robust(tmp_path: Path):
    records = []
    for index in range(20):
        records.append(
            build_auxiliary_record(
                context=((0.1,) * 8) if index % 2 == 0 else ((0.9,) * 8),
                skill_id=1 if index % 2 == 0 else 2,
                payoff=1.7 if index % 2 == 0 else 1.1,
                motives=(0.8, 0.4) if index % 2 == 0 else (0.3, 0.7),
                baseline_stats=_baseline_stats(),
                motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
                record_behavior_probability=0.5,
                all_candidate_delta_r=(0.4, 0.2),
                all_candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
                selected_candidate_index=0,
            )
        )

    result = train_auxiliary_from_records(
        records,
        checkpoint_path=str(tmp_path / "mdn_auxiliary_dr_best.pth"),
        seed=0,
        device="cpu",
        use_doubly_robust=True,
    )

    assert Path(result["checkpoint_path"]).exists()
    assert result["best_val_loss"] >= 0.0


def test_probability_aware_path_recomputes_softmax_target_probability_from_candidate_scores():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    probability = trainer._compute_softmax_target_probability(
        selected_index=0,
        candidate_delta_r=(0.4, 0.2),
        candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
        weights=np.array([0.9, 0.1], dtype=np.float32),
    )

    assert 0.5 < probability < 1.0


def test_probability_aware_training_requires_candidate_score_fields():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    record = build_auxiliary_record(
        context=(0.1,) * 8,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
        behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
        target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        record_behavior_probability=0.5,
        use_ips=True,
    )

    with pytest.raises(ValueError, match="candidate_delta_r"):
        trainer.train_probability_aware_records([record])


def test_probability_aware_loss_weights_q_loss_instead_of_scaling_target():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(model, config=MDNAuxiliaryTrainerConfig(), device="cpu")

    gate_logits = torch.tensor([0.0], dtype=torch.float32)
    q_hat = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    accept_label = torch.tensor([1.0], dtype=torch.float32)
    q_target = torch.tensor([[1.0, 2.0]], dtype=torch.float32)

    weighted_total_loss, _, weighted_q_loss = trainer._compute_losses(
        gate_logits,
        q_hat,
        accept_label,
        q_target,
        q_loss_weight=3.0,
    )
    unweighted_total_loss, _, unweighted_q_loss = trainer._compute_losses(
        gate_logits,
        q_hat,
        accept_label,
        q_target,
        q_loss_weight=1.0,
    )

    assert torch.isclose(weighted_q_loss, unweighted_q_loss)
    expected_delta = trainer.config.lambda_q * 2.0 * unweighted_q_loss
    actual_delta = weighted_total_loss - unweighted_total_loss
    assert torch.isclose(actual_delta, expected_delta)


def _probability_aware_record() -> object:
    return build_auxiliary_record(
        context=(0.1,) * 8,
        skill_id=1,
        payoff=1.2,
        motives=(0.5, 0.2),
        baseline_stats=_baseline_stats(),
        motive_trajectory=[[(1.0, 0.0), (0.0, 1.0)]],
        behavior_probability=np.array([[0.5, 0.5]], dtype=np.float32),
        target_probability=np.array([[1.0, 1.0]], dtype=np.float32),
        record_behavior_probability=0.5,
        use_ips=True,
        all_candidate_delta_r=(0.4, 0.2),
        all_candidate_delta_n=((0.8, 0.1), (0.1, 0.8)),
        selected_candidate_index=0,
    )


def test_dr_mode_runs_and_loss_is_finite():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    metrics = trainer._run_probability_aware_epoch([_probability_aware_record()], training=True)

    for value in metrics.values():
        assert np.isfinite(value)


def test_ips_mode_unchanged_by_dr_flag():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, use_doubly_robust=False, max_epochs=1, batch_size=1),
        device="cpu",
    )

    metrics = trainer._run_probability_aware_epoch([_probability_aware_record()], training=True)

    for value in metrics.values():
        assert np.isfinite(value)


def test_dr_and_ips_modes_are_independently_selectable():
    record = _probability_aware_record()
    model_ips = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer_ips = MDNAuxiliaryTrainer(
        model_ips,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, use_doubly_robust=False, max_epochs=1, batch_size=1),
        device="cpu",
    )
    ips_metrics = trainer_ips._run_probability_aware_epoch([record], training=True)

    model_dr = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer_dr = MDNAuxiliaryTrainer(
        model_dr,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, max_epochs=1, batch_size=1),
        device="cpu",
    )
    dr_metrics = trainer_dr._run_probability_aware_epoch([record], training=True)

    assert np.isfinite(ips_metrics["loss"])
    assert np.isfinite(dr_metrics["loss"])


def test_dr_baseline_has_no_gradient():
    q_hat = torch.tensor([[0.2, 0.4]], dtype=torch.float32, requires_grad=True)
    baseline = q_hat.detach()
    dr_target = baseline + 1.5 * (torch.tensor([[1.0, 2.0]], dtype=torch.float32) - baseline)
    loss = torch.nn.functional.mse_loss(q_hat, dr_target)
    loss.backward()

    assert baseline.grad is None
    assert q_hat.grad is not None


def test_dr_single_record_does_not_crash():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    metrics = trainer.online_step(_probability_aware_record())

    assert np.isfinite(metrics["loss"])


def test_dr_seeded_training_remains_stable():
    torch.manual_seed(0)
    np.random.seed(0)
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_doubly_robust=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    for _ in range(10):
        metrics = trainer.online_step(_probability_aware_record())
        assert np.isfinite(metrics["loss"])


def test_gate_only_probability_aware_record_skips_q_loss():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(
        model,
        config=MDNAuxiliaryTrainerConfig(use_ips=True, max_epochs=1, batch_size=1),
        device="cpu",
    )

    record = _probability_aware_record()
    record.has_q_target = False
    record.q_target = (0.0, 0.0)

    metrics = trainer._run_probability_aware_epoch([record], training=True)

    assert np.isfinite(metrics["loss"])
    assert metrics["q_loss"] == pytest.approx(0.0)


def test_ips_and_dr_modes_are_mutually_exclusive():
    model = MotiveDecompositionNetwork(input_dim=8, num_skills=4, num_objectives=2)

    with pytest.raises(ValueError, match="mutually exclusive"):
        MDNAuxiliaryTrainer(
            model,
            config=MDNAuxiliaryTrainerConfig(use_ips=True, use_doubly_robust=True),
            device="cpu",
        )
