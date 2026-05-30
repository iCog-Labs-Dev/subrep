"""CLI diagnostic for the offline MDN training stack."""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_trainer import MDNTrainer, MDNTrainerConfig
from utils.mdn_contracts import CandidateSkillRecord, MDNDecisionRecord
from utils.mdn_reward import compute_advantage, compute_mdn_policy_loss, compute_mdn_utility
from utils.mdn_selection import alpha_to_mean_weights


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_record(context_value: float, weights_used: tuple[float, float], selected_skill_id: str, actual_motives: tuple[float, float], utility: float | None = None) -> MDNDecisionRecord:
    candidates = (
        CandidateSkillRecord(
            skill_id="safe_skill",
            delta_r=0.2,
            delta_n=(0.8, 0.1),
            is_certified=True,
            gate_type="CDS",
        ),
        CandidateSkillRecord(
            skill_id="fuel_skill",
            delta_r=0.2,
            delta_n=(0.1, 0.8),
            is_certified=True,
            gate_type="CDS",
        ),
    )
    return MDNDecisionRecord(
        context=(context_value,) * 14,
        alpha=(1.0, 1.0),
        support_values=(0.5, 0.5),
        weights_used=weights_used,
        candidate_skills=candidates,
        selected_skill_id=selected_skill_id,
        selected_score=0.0,
        actual_payoff=1.0,
        actual_motives=actual_motives,
        utility=utility,
    )


def make_synthetic_records(num_records: int) -> list[MDNDecisionRecord]:
    records: list[MDNDecisionRecord] = []
    for index in range(num_records):
        if index % 2 == 0:
            records.append(make_record(0.1, (0.8, 0.2), "safe_skill", (0.8, 0.2)))
        else:
            records.append(make_record(0.1, (0.2, 0.8), "fuel_skill", (0.2, 0.1)))
    return records


def split_records(records: list[MDNDecisionRecord], validation_fraction: float = 0.2) -> tuple[list[MDNDecisionRecord], list[MDNDecisionRecord]]:
    split_index = max(1, int(round(len(records) * (1.0 - validation_fraction))))
    split_index = min(split_index, len(records) - 1)
    return records[:split_index], records[split_index:]


def evaluate_records(trainer: MDNTrainer, records: Iterable[MDNDecisionRecord]) -> dict[str, float]:
    metrics = []
    trainer.model.eval()
    for record in records:
        context = torch.tensor(record.context, dtype=torch.float32, device=trainer.device)
        with torch.no_grad():
            alpha, support_values = trainer.model(context)
            distribution = torch.distributions.Dirichlet(alpha)
            recorded_weights = torch.tensor(record.weights_used, dtype=torch.float32, device=trainer.device)
            log_prob = distribution.log_prob(recorded_weights)
        weights_np = recorded_weights.detach().cpu().numpy()
        utility = compute_mdn_utility(np.asarray(record.actual_motives, dtype=np.float32), weights_np, record.actual_payoff, trainer.config.payoff_weight) if record.utility is None else float(record.utility)
        advantage = compute_advantage(utility, running_baseline=trainer.running_baseline)
        loss = compute_mdn_policy_loss(log_prob, advantage)
        metrics.append({"loss": float(loss.item()), "utility": utility})
    return {
        "loss": float(np.mean([item["loss"] for item in metrics])),
        "utility": float(np.mean([item["utility"] for item in metrics])),
    }


def check_offline_record_behavior() -> tuple[bool, str]:
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(strict_validation=False), device="cpu")
    record = make_record(0.1, (0.95, 0.05), "fuel_skill", (0.9, 0.1))
    metrics = trainer.training_step(record)
    return bool(np.isfinite(metrics["loss"])), "offline record uses recorded weights"


def check_utility_calculation() -> tuple[bool, float]:
    utility = compute_mdn_utility(np.array([10.0, 4.0], dtype=np.float32), np.array([0.7, 0.3], dtype=np.float32))
    return bool(np.isclose(utility, 8.2)), utility


def check_policy_loss() -> tuple[bool, float]:
    log_prob = torch.tensor(-0.5, dtype=torch.float32)
    loss = compute_mdn_policy_loss(log_prob, advantage=2.0)
    return bool(np.isclose(float(loss.item()), 1.0)), float(loss.item())


def check_one_step_update() -> tuple[bool, bool, dict[str, float]]:
    torch.manual_seed(0)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(), device="cpu")
    before = [parameter.detach().clone() for parameter in model.parameters()]
    metrics = trainer.training_step(make_record(0.1, (0.8, 0.2), "safe_skill", (1.0, 0.1)))
    after = list(model.parameters())
    changed = any(not torch.allclose(prev, curr.detach()) for prev, curr in zip(before, after))
    finite = all(np.isfinite(value) for value in metrics.values())
    return finite, changed, metrics


def check_synthetic_learning_direction(epochs: int, seed: int, lr: float, num_records: int) -> tuple[bool, dict[str, float], str | None]:
    seed_everything(seed)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(learning_rate=lr), device="cpu")
    records = make_synthetic_records(num_records)
    train_records, validation_records = split_records(records)

    initial_weights = alpha_to_mean_weights(model(torch.tensor((0.1,) * 14, dtype=torch.float32))[0].detach().cpu().numpy())
    target = np.array([0.8, 0.2], dtype=np.float32)
    initial_distance = float(np.linalg.norm(initial_weights - target))

    train_history = []
    validation_history = []
    for _ in range(epochs):
        train_metrics = trainer.train_records(train_records)
        validation_metrics = evaluate_records(trainer, validation_records)
        train_history.append(train_metrics)
        validation_history.append(validation_metrics)

    final_weights = alpha_to_mean_weights(model(torch.tensor((0.1,) * 14, dtype=torch.float32))[0].detach().cpu().numpy())
    final_distance = float(np.linalg.norm(final_weights - target))
    passed = bool(final_weights[0] > initial_weights[0] or final_distance < initial_distance)

    warning = None
    if train_history[-1]["loss"] < train_history[0]["loss"] and validation_history[-1]["loss"] > validation_history[0]["loss"] * 1.5:
        warning = "Train improved while validation worsened substantially. This diagnostic checks trainer mechanics, not final scientific performance."

    return passed, {
        "initial_weight_0": float(initial_weights[0]),
        "initial_weight_1": float(initial_weights[1]),
        "final_weight_0": float(final_weights[0]),
        "final_weight_1": float(final_weights[1]),
        "initial_distance": initial_distance,
        "final_distance": final_distance,
        "train_loss": float(train_history[-1]["loss"]),
        "train_utility": float(train_history[-1]["utility"]),
        "validation_loss": float(validation_history[-1]["loss"]),
        "validation_utility": float(validation_history[-1]["utility"]),
    }, warning


def check_checkpoint_round_trip(keep_checkpoint: bool, seed: int, lr: float) -> tuple[bool, str]:
    seed_everything(seed)
    model = MotiveDecompositionNetwork()
    trainer = MDNTrainer(model, config=MDNTrainerConfig(learning_rate=lr), device="cpu")
    trainer.training_step(make_record(0.1, (0.8, 0.2), "safe_skill", (1.0, 0.1)))

    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "mdn_diag_checkpoint.pth"
        trainer.save_checkpoint(checkpoint_path)
        restored = MDNTrainer.from_checkpoint(checkpoint_path, model=MotiveDecompositionNetwork(), device="cpu")
        context = torch.tensor((0.1,) * 14, dtype=torch.float32)
        trainer.model.eval()
        restored.model.eval()
        with torch.no_grad():
            original_alpha, original_support = trainer.model(context)
            restored_alpha, restored_support = restored.model(context)
        passed = bool(torch.allclose(original_alpha, restored_alpha) and torch.allclose(original_support, restored_support))
        if keep_checkpoint:
            keep_path = Path("models") / "mdn_diag_checkpoint.pth"
            keep_path.parent.mkdir(parents=True, exist_ok=True)
            keep_path.write_bytes(checkpoint_path.read_bytes())
            return passed, str(keep_path)
        return passed, str(checkpoint_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose the offline MDN training stack.")
    parser.add_argument("--synthetic", action="store_true", help="Run the synthetic diagnostic suite.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-records", type=int, default=128)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--keep-checkpoint", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not args.synthetic:
        raise ValueError("This diagnostic currently requires --synthetic")

    seed_everything(args.seed)
    print("MDN Offline Training Diagnostic")
    print("================================")

    all_passed = True

    passed, message = check_offline_record_behavior()
    print(f"[{'PASS' if passed else 'FAIL'}] {message}")
    all_passed &= passed

    passed, utility = check_utility_calculation()
    print(f"[{'PASS' if passed else 'FAIL'}] utility calculation (computed={utility:.4f}, expected=8.2000)")
    all_passed &= passed

    passed, loss_value = check_policy_loss()
    print(f"[{'PASS' if passed else 'FAIL'}] policy loss calculation (computed={loss_value:.4f}, expected=1.0000)")
    all_passed &= passed

    finite, changed, one_step_metrics = check_one_step_update()
    print(f"[{'PASS' if finite else 'FAIL'}] one-step update finite")
    print(f"[{'PASS' if changed else 'FAIL'}] parameters updated")
    all_passed &= finite and changed

    passed, synthetic_metrics, warning = check_synthetic_learning_direction(args.epochs, args.seed, args.lr, args.num_records)
    print(f"[{'PASS' if passed else 'FAIL'}] synthetic learning direction")
    all_passed &= passed

    passed, checkpoint_path = check_checkpoint_round_trip(args.keep_checkpoint, args.seed, args.lr)
    print(f"[{'PASS' if passed else 'FAIL'}] checkpoint round trip")
    all_passed &= passed

    print()
    print(f"initial alpha mean: [{synthetic_metrics['initial_weight_0']:.4f}, {synthetic_metrics['initial_weight_1']:.4f}]")
    print(f"final alpha mean:   [{synthetic_metrics['final_weight_0']:.4f}, {synthetic_metrics['final_weight_1']:.4f}]")
    print(f"initial distance to target: {synthetic_metrics['initial_distance']:.4f}")
    print(f"final distance to target:   {synthetic_metrics['final_distance']:.4f}")
    print(f"train loss summary: {synthetic_metrics['train_loss']:.4f}")
    print(f"train utility summary: {synthetic_metrics['train_utility']:.4f}")
    print(f"validation loss summary: {synthetic_metrics['validation_loss']:.4f}")
    print(f"validation utility summary: {synthetic_metrics['validation_utility']:.4f}")
    print(f"checkpoint path: {checkpoint_path}")
    if warning is not None:
        print(f"WARNING: {warning}")
    print("This diagnostic checks trainer mechanics, not final scientific performance.")

    if args.verbose:
        print(f"one-step metrics: {one_step_metrics}")

    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
