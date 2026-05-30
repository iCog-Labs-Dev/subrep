"""CLI diagnostic for the auxiliary proposal-conditioned MDN training path."""

from __future__ import annotations

import argparse
import random
import tempfile
from pathlib import Path

import numpy as np
import torch

from generator.mdn import MotiveDecompositionNetwork
from generator.mdn_auxiliary_trainer import MDNAuxiliaryTrainer, MDNAuxiliaryTrainerConfig, build_auxiliary_record
from generator.train_mdn_auxiliary import train_auxiliary_from_records


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def baseline_stats() -> dict[str, object]:
    return {
        "baseline_payoff": 1.0,
        "baseline_motives": (0.5, 0.2),
    }


def make_aux_record(context_value: float, skill_id: int, payoff: float, motives: tuple[float, float], use_ips: bool = False):
    kwargs = {}
    if use_ips:
        kwargs = {
            "motive_trajectory": [[(1.0, 0.0), (0.0, 1.0)]],
            "behavior_probability": np.array([[0.5, 0.5]], dtype=np.float32),
            "target_probability": np.array([[1.0, 1.0]], dtype=np.float32),
            "record_behavior_probability": 0.5,
            "use_ips": True,
        }
    return build_auxiliary_record(
        context=(context_value,) * 14,
        skill_id=skill_id,
        payoff=payoff,
        motives=motives,
        baseline_stats=baseline_stats(),
        **kwargs,
    )


def make_scenario_records(mode: str, num_records: int, use_ips: bool = False):
    records = []
    for index in range(num_records):
        if mode == "safety":
            records.append(make_aux_record(0.1, 1, 1.7, (0.9, 0.1), use_ips=use_ips))
        elif mode == "fuel":
            records.append(make_aux_record(0.9, 2, 1.1, (0.1, 0.9), use_ips=use_ips))
        else:
            if index % 2 == 0:
                records.append(make_aux_record(0.5, 1, 1.4, (0.6, 0.5), use_ips=use_ips))
            else:
                records.append(make_aux_record(0.5, 2, 1.4, (0.5, 0.6), use_ips=use_ips))
    return records


def initial_outputs(seed: int):
    seed_everything(seed)
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=8, num_objectives=2)
    model.eval()
    with torch.no_grad():
        safety_gate, safety_q = model.forward_auxiliary(torch.tensor((0.1,) * 14, dtype=torch.float32), torch.tensor(1))
        fuel_gate, fuel_q = model.forward_auxiliary(torch.tensor((0.9,) * 14, dtype=torch.float32), torch.tensor(2))
        balanced_gate, balanced_q = model.forward_auxiliary(torch.tensor((0.5,) * 14, dtype=torch.float32), torch.tensor(1))
    return {
        "safety_gate": float(torch.sigmoid(safety_gate).item()),
        "fuel_gate": float(torch.sigmoid(fuel_gate).item()),
        "balanced_gate": float(torch.sigmoid(balanced_gate).item()),
        "safety_q": tuple(float(v) for v in safety_q.tolist()),
        "fuel_q": tuple(float(v) for v in fuel_q.tolist()),
        "balanced_q": tuple(float(v) for v in balanced_q.tolist()),
    }


def checkpoint_round_trip(seed: int):
    seed_everything(seed)
    model = MotiveDecompositionNetwork(input_dim=14, num_skills=8, num_objectives=2)
    trainer = MDNAuxiliaryTrainer(model, config=MDNAuxiliaryTrainerConfig(), device="cpu")
    trainer.train_records(make_scenario_records("safety", 8))
    with tempfile.TemporaryDirectory() as temp_dir:
        checkpoint_path = Path(temp_dir) / "mdn_aux_diag.pt"
        trainer_state = trainer.model.state_dict()
        torch.save({"model_state_dict": trainer_state}, checkpoint_path)
        restored = MotiveDecompositionNetwork(input_dim=14, num_skills=8, num_objectives=2)
        restored.load_state_dict(torch.load(checkpoint_path, map_location="cpu")["model_state_dict"])
        with torch.no_grad():
            original = trainer.model.forward_auxiliary(torch.tensor((0.1,) * 14, dtype=torch.float32), torch.tensor(1))
            loaded = restored.forward_auxiliary(torch.tensor((0.1,) * 14, dtype=torch.float32), torch.tensor(1))
        return bool(torch.allclose(original[0], loaded[0]) and torch.allclose(original[1], loaded[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose auxiliary MDN training.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-records", type=int, default=64)
    parser.add_argument("--use-ips", action="store_true")
    args = parser.parse_args()

    print("MDN Auxiliary Training Diagnostic")
    print("=================================")

    initial = initial_outputs(args.seed)
    print(f"Initial safety gate prob: {initial['safety_gate']:.4f}")
    print(f"Initial fuel gate prob:   {initial['fuel_gate']:.4f}")
    print(f"Initial balanced gate prob: {initial['balanced_gate']:.4f}")
    print(f"Initial safety q: {initial['safety_q']}")
    print(f"Initial fuel q:   {initial['fuel_q']}")
    print(f"Initial balanced q: {initial['balanced_q']}")

    records = make_scenario_records("safety", args.num_records // 3, use_ips=args.use_ips)
    records += make_scenario_records("fuel", args.num_records // 3, use_ips=args.use_ips)
    records += make_scenario_records("balanced", args.num_records - 2 * (args.num_records // 3), use_ips=args.use_ips)

    with tempfile.TemporaryDirectory() as temp_dir:
        result = train_auxiliary_from_records(
            records,
            checkpoint_path=str(Path(temp_dir) / "mdn_auxiliary_best.pth"),
            seed=args.seed,
            device="cpu",
            use_ips=args.use_ips,
        )

    print(f"Train loss: {result['best_metrics']['train']['loss']:.4f}")
    print(f"Train gate accuracy: {result['best_metrics']['train']['gate_accuracy']:.4f}")
    print(f"Train q loss: {result['best_metrics']['train']['q_loss']:.4f}")
    print(f"Validation loss: {result['best_metrics']['val']['loss']:.4f}")
    print(f"Validation gate accuracy: {result['best_metrics']['val']['gate_accuracy']:.4f}")
    print(f"Validation q loss: {result['best_metrics']['val']['q_loss']:.4f}")

    ok_checkpoint = checkpoint_round_trip(args.seed)
    print(f"[{'PASS' if ok_checkpoint else 'FAIL'}] checkpoint round trip")
    print("This diagnostic checks auxiliary trainer mechanics and proposal-conditioned learning behavior, not final scientific performance.")

    return 0 if ok_checkpoint else 1


if __name__ == "__main__":
    raise SystemExit(main())
