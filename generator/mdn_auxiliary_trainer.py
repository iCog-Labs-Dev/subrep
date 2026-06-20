"""Auxiliary proposal-conditioned trainer for the MDN shared representation path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import torch
from torch.nn import BCEWithLogitsLoss, MSELoss
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, random_split

from baseline.improvement_calculator import ImprovementCalculator
from certification.cds_test import CDSGate
from certification.pds_test import PDSGate
from generator.mdn import MotiveDecompositionNetwork
from utils.mdn_selection import alpha_to_mean_weights
from utils.return_targets import discounted_motive_return, doubly_robust_return
from utils.weight_set_store import WeightSet


def _seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AuxiliaryTrainingRecord:
    context: tuple[float, ...]
    skill_id: int
    accept_label: float
    q_target: tuple[float, ...]
    has_q_target: bool = True
    behavior_probability: float | None = None
    motive_trajectory: tuple[tuple[float, ...], ...] | None = None
    # Delta info for ALL certified candidates at data-collection time.
    # Stored in the same order; selected_candidate_index points to the chosen skill.
    # Required to recompute target_probability from current MDN weights at training time.
    candidate_delta_r: tuple[float, ...] | None = None
    candidate_delta_n: tuple[tuple[float, ...], ...] | None = None
    selected_candidate_index: int | None = None


class AuxiliaryDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, records: Iterable[AuxiliaryTrainingRecord]) -> None:
        self.records = list(records)
        if not self.records:
            raise ValueError("AuxiliaryDataset requires at least one record")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        record = self.records[index]
        return (
            torch.tensor(record.context, dtype=torch.float32),
            torch.tensor(record.skill_id, dtype=torch.long),
            torch.tensor(record.accept_label, dtype=torch.float32),
            torch.tensor(record.q_target, dtype=torch.float32),
        )


@dataclass
class MDNAuxiliaryTrainerConfig:
    lambda_q: float = 0.1
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 16
    validation_split: float = 0.2
    max_epochs: int = 50
    patience: int = 5
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    gradient_clip_norm: float = 1.0
    checkpoint_path: str = "models/mdn_auxiliary_best.pth"
    random_seed: int = 0
    use_ips: bool = False
    ips_clip: float = 10.0
    use_doubly_robust: bool = False


class MDNAuxiliaryTrainer:
    """Train the proposal-conditioned auxiliary MDN model with BCE + MSE."""

    def __init__(self, model: MotiveDecompositionNetwork, config: Optional[MDNAuxiliaryTrainerConfig] = None, device: Optional[str] = None) -> None:
        self.model = model
        self.config = config or MDNAuxiliaryTrainerConfig()
        if self.config.use_ips and self.config.use_doubly_robust:
            raise ValueError("use_ips and use_doubly_robust are mutually exclusive auxiliary estimators")
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.gate_loss_fn = BCEWithLogitsLoss()
        self.q_loss_fn = MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=self.config.scheduler_factor,
            patience=self.config.scheduler_patience,
        )

    def _compute_losses(
        self,
        gate_logits: torch.Tensor,
        q_hat: torch.Tensor,
        accept_label: torch.Tensor,
        q_target: torch.Tensor,
        q_loss_weight: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_loss = self.gate_loss_fn(gate_logits, accept_label)
        q_loss = self.q_loss_fn(q_hat, q_target)
        total_loss = gate_loss + self.config.lambda_q * float(q_loss_weight) * q_loss
        return total_loss, gate_loss, q_loss

    def _compute_gate_only_losses(
        self,
        gate_logits: torch.Tensor,
        q_hat: torch.Tensor,
        accept_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate_loss = self.gate_loss_fn(gate_logits, accept_label)
        q_loss = torch.zeros((), dtype=q_hat.dtype, device=q_hat.device)
        return gate_loss, gate_loss, q_loss

    def _compute_softmax_target_probability(
        self,
        selected_index: int,
        candidate_delta_r: tuple[float, ...],
        candidate_delta_n: tuple[tuple[float, ...], ...],
        weights: np.ndarray,
    ) -> float:
        """Recompute softmax selection probability for the selected skill using current MDN weights.
        Returns the softmax probability of the candidate at selected_index.
        """
        if not candidate_delta_r or not candidate_delta_n:
            raise ValueError("candidate_delta_r and candidate_delta_n must not be empty")
        if len(candidate_delta_r) != len(candidate_delta_n):
            raise ValueError("candidate_delta_r and candidate_delta_n must have the same length")
        if selected_index < 0 or selected_index >= len(candidate_delta_r):
            raise ValueError("selected_index must refer to a candidate entry")

        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        scores = np.array(
            [float(dr) + float(np.dot(weights, np.asarray(dn, dtype=np.float64)))
             for dr, dn in zip(candidate_delta_r, candidate_delta_n)],
            dtype=np.float64,
        )
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        return float(exp_scores[selected_index] / np.sum(exp_scores))

    def _run_epoch(self, loader: DataLoader, training: bool) -> dict[str, float]:
        self.model.train(training)
        total_loss = 0.0
        total_gate_loss = 0.0
        total_q_loss = 0.0
        total_correct = 0.0
        total_examples = 0

        for context, skill_id, accept_label, q_target in loader:
            context = context.to(self.device)
            skill_id = skill_id.to(self.device)
            accept_label = accept_label.to(self.device)
            q_target = q_target.to(self.device)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            gate_logits, q_hat = self.model.forward_auxiliary(context, skill_id)
            if self.config.use_ips:
                raise ValueError(
                    "use_ips=True requires train_probability_aware_records(...), not train_records(...)"
                )
            total_batch_loss, gate_loss, q_loss = self._compute_losses(gate_logits, q_hat, accept_label, q_target)

            if training:
                total_batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()

            batch_size = context.shape[0]
            predictions = (torch.sigmoid(gate_logits) >= 0.5).float()
            total_correct += float((predictions == accept_label).sum().item())
            total_examples += batch_size
            total_loss += float(total_batch_loss.item()) * batch_size
            total_gate_loss += float(gate_loss.item()) * batch_size
            total_q_loss += float(q_loss.item()) * batch_size

        return {
            "loss": total_loss / total_examples,
            "gate_loss": total_gate_loss / total_examples,
            "q_loss": total_q_loss / total_examples,
            "gate_accuracy": total_correct / total_examples,
        }

    def _run_probability_aware_epoch(
        self, records: list[AuxiliaryTrainingRecord], training: bool
    ) -> dict[str, float]:
        """Run one epoch of IPS-corrected auxiliary training over a list of records.
        """
        self.model.train(training)
        total_loss = 0.0
        total_gate_loss = 0.0
        total_q_loss = 0.0
        total_correct = 0.0

        for record in records:
            if record.behavior_probability is None:
                raise ValueError("probability-aware records must have behavior_probability")
            if record.candidate_delta_r is None:
                raise ValueError("probability-aware records must have candidate_delta_r")
            if record.candidate_delta_n is None:
                raise ValueError("probability-aware records must have candidate_delta_n")
            if record.selected_candidate_index is None:
                raise ValueError("probability-aware records must have selected_candidate_index")

            ctx = torch.tensor(record.context, dtype=torch.float32, device=self.device).unsqueeze(0)
            sid = torch.tensor([record.skill_id], dtype=torch.long, device=self.device)
            label = torch.tensor([record.accept_label], dtype=torch.float32, device=self.device)
            q_tgt = torch.tensor(record.q_target, dtype=torch.float32, device=self.device).unsqueeze(0)

            if training:
                self.optimizer.zero_grad(set_to_none=True)

            gate_logits, q_hat = self.model.forward_auxiliary(ctx, sid)

            if not record.has_q_target:
                batch_loss, gate_loss, q_loss = self._compute_gate_only_losses(
                    gate_logits,
                    q_hat,
                    label,
                )
            else:
                alpha, _ = self.model.forward_inference(ctx)
                weights = alpha_to_mean_weights(alpha.detach().squeeze(0).cpu().numpy())
                target_prob = self._compute_softmax_target_probability(
                    record.selected_candidate_index,
                    record.candidate_delta_r,
                    record.candidate_delta_n,
                    weights,
                )
                raw_weight = target_prob / max(float(record.behavior_probability), 1e-8)
                ips_weight = min(raw_weight, float(self.config.ips_clip))

                if self.config.use_doubly_robust:
                    baseline = q_hat.detach()
                    dr_target = baseline + ips_weight * (q_tgt - baseline)
                    batch_loss, gate_loss, q_loss = self._compute_losses(
                        gate_logits,
                        q_hat,
                        label,
                        dr_target,
                        q_loss_weight=1.0,
                    )
                else:
                    # IPS is a loss weight, not target scaling. Scaling q_tgt by
                    # rho would train q_hat toward rho * return and can inflate
                    # values; weighting the loss estimates target-policy risk.
                    batch_loss, gate_loss, q_loss = self._compute_losses(
                        gate_logits,
                        q_hat,
                        label,
                        q_tgt,
                        q_loss_weight=ips_weight,
                    )

            if training:
                batch_loss.backward()
                clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()

            pred = (torch.sigmoid(gate_logits) >= 0.5).float()
            total_correct += float((pred == label).sum().item())
            total_loss += float(batch_loss.item())
            total_gate_loss += float(gate_loss.item())
            total_q_loss += float(q_loss.item())

        n = len(records)
        return {
            "loss": total_loss / n,
            "gate_loss": total_gate_loss / n,
            "q_loss": total_q_loss / n,
            "gate_accuracy": total_correct / n,
        }

    def online_step(self, record: AuxiliaryTrainingRecord) -> dict[str, float]:
        """Run one online gradient step on a single auxiliary training record.
        """
        if self.config.use_ips or self.config.use_doubly_robust:
            if record.behavior_probability is None:
                raise ValueError("probability-aware online_step requires behavior_probability")
            if record.candidate_delta_r is None:
                raise ValueError("probability-aware online_step requires candidate_delta_r")
            if record.candidate_delta_n is None:
                raise ValueError("probability-aware online_step requires candidate_delta_n")
            if record.selected_candidate_index is None:
                raise ValueError("probability-aware online_step requires selected_candidate_index")
            return self._run_probability_aware_epoch([record], training=True)

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        ctx = torch.tensor(record.context, dtype=torch.float32, device=self.device).unsqueeze(0)
        sid = torch.tensor([record.skill_id], dtype=torch.long, device=self.device)
        label = torch.tensor([record.accept_label], dtype=torch.float32, device=self.device)
        q_tgt = torch.tensor(record.q_target, dtype=torch.float32, device=self.device).unsqueeze(0)
        gate_logits, q_hat = self.model.forward_auxiliary(ctx, sid)
        total_loss, gate_loss, q_loss = self._compute_losses(gate_logits, q_hat, label, q_tgt)
        total_loss.backward()
        clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        pred = (torch.sigmoid(gate_logits) >= 0.5).float()
        return {
            "loss": float(total_loss.item()),
            "gate_loss": float(gate_loss.item()),
            "q_loss": float(q_loss.item()),
            "gate_accuracy": float((pred == label).float().mean().item()),
        }

    def train_records(self, records: Iterable[AuxiliaryTrainingRecord]) -> dict[str, Any]:
        if self.config.use_ips or self.config.use_doubly_robust:
            raise ValueError(
                "probability-aware auxiliary estimators require train_probability_aware_records(...), not train_records(...)"
            )
        dataset = AuxiliaryDataset(records)
        val_size = max(1, int(round(len(dataset) * self.config.validation_split)))
        train_size = len(dataset) - val_size
        if train_size <= 0:
            raise ValueError("Dataset too small for configured validation split")

        generator = torch.Generator().manual_seed(self.config.random_seed)
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, generator=generator)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False)

        best_val_loss = float("inf")
        best_metrics: dict[str, Any] = {}
        epochs_without_improvement = 0

        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_epoch(train_loader, training=True)
            with torch.no_grad():
                val_metrics = self._run_epoch(val_loader, training=False)

            self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                best_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "config": self.config.__dict__,
                        "metrics": best_metrics,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        return {
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "checkpoint_path": str(checkpoint_path),
        }

    def train_probability_aware_records(self, records: Iterable[AuxiliaryTrainingRecord]) -> dict[str, Any]:
        """Train with IPS correction using softmax-over-scores selection probabilities.
        """
        records_list = list(records)
        if not records_list:
            raise ValueError("train_probability_aware_records requires at least one record")
        for record in records_list:
            if record.behavior_probability is None:
                raise ValueError("All probability-aware records must include behavior_probability")
            if record.candidate_delta_r is None:
                raise ValueError("All probability-aware records must include candidate_delta_r")
            if record.candidate_delta_n is None:
                raise ValueError("All probability-aware records must include candidate_delta_n")
            if record.selected_candidate_index is None:
                raise ValueError("All probability-aware records must include selected_candidate_index")

        rng = np.random.default_rng(self.config.random_seed)
        indices = rng.permutation(len(records_list))
        val_size = max(1, int(round(len(records_list) * self.config.validation_split)))
        train_size = len(records_list) - val_size
        if train_size <= 0:
            raise ValueError("Dataset too small for configured validation split")

        train_records = [records_list[int(i)] for i in indices[:train_size]]
        val_records = [records_list[int(i)] for i in indices[train_size:]]

        best_val_loss = float("inf")
        best_metrics: dict[str, Any] = {}
        epochs_without_improvement = 0

        checkpoint_path = Path(self.config.checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.max_epochs + 1):
            train_metrics = self._run_probability_aware_epoch(train_records, training=True)
            with torch.no_grad():
                val_metrics = self._run_probability_aware_epoch(val_records, training=False)

            self.scheduler.step(val_metrics["loss"])

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                epochs_without_improvement = 0
                best_metrics = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "config": self.config.__dict__,
                        "metrics": best_metrics,
                    },
                    checkpoint_path,
                )
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.config.patience:
                break

        return {
            "best_val_loss": best_val_loss,
            "best_metrics": best_metrics,
            "checkpoint_path": str(checkpoint_path),
        }


def build_auxiliary_record(
    *,
    context,
    skill_id: int,
    payoff: float,
    motives,
    baseline_stats: Optional[dict[str, Any]] = None,
    accept_label: Optional[bool] = None,
    gate_type: str = "CDS",
    epsilon: Optional[float] = None,
    motive_trajectory: Optional[np.ndarray] = None,
    behavior_probability: Optional[np.ndarray] = None,
    target_probability: Optional[np.ndarray] = None,
    q_model_estimate: Optional[np.ndarray] = None,
    record_behavior_probability: Optional[float] = None,
    use_ips: bool = False,
    use_doubly_robust: bool = False,
    gamma: float = 1.0,
    weight_set: Optional[WeightSet] = None,
    all_candidate_delta_r: Optional[tuple[float, ...]] = None,
    all_candidate_delta_n: Optional[tuple[tuple[float, ...], ...]] = None,
    selected_candidate_index: Optional[int] = None,
) -> AuxiliaryTrainingRecord:

    context = tuple(float(v) for v in np.asarray(context, dtype=np.float32).reshape(-1))

    if accept_label is None:
        if baseline_stats is None:
            raise ValueError("baseline_stats is required when accept_label is not provided")
        calculator = ImprovementCalculator(baseline_stats)
        delta_r, delta_n = calculator.compute_improvements(skill_payoff=payoff, skill_motives=motives)
        gate_type_normalized = gate_type.strip().upper()
        if gate_type_normalized == "CDS":
            accept_label = CDSGate().admit(delta_r, delta_n, weight_set=weight_set)
        elif gate_type_normalized == "PDS":
            accept_label = PDSGate(epsilon=0.1 if epsilon is None else float(epsilon)).admit(delta_r, delta_n, weight_set=weight_set)
        else:
            raise ValueError(f"gate_type must be 'CDS' or 'PDS', got {gate_type!r}")

    if use_ips and use_doubly_robust:
        raise ValueError("use_ips and use_doubly_robust cannot both be True")

    if motive_trajectory is None:
        q_target = np.asarray(motives, dtype=np.float32).reshape(-1)
    elif use_doubly_robust:
        q_target = doubly_robust_return(
            np.asarray(motive_trajectory, dtype=np.float32),
            behavior_probability=behavior_probability,
            target_probability=target_probability,
            q_model_estimate=q_model_estimate,
            gamma=gamma,
        )
    elif use_ips:
        if behavior_probability is None or target_probability is None:
            raise ValueError("IPS return requires behavior_probability and target_probability")
        # Probability-aware records keep the plain discounted target.
        # The trainer applies the IPS ratio as a sample weight to avoid double weighting.
        q_target = discounted_motive_return(np.asarray(motive_trajectory, dtype=np.float32), gamma=gamma)
    else:
        q_target = discounted_motive_return(np.asarray(motive_trajectory, dtype=np.float32), gamma=gamma)

    q_target = np.asarray(q_target, dtype=np.float32).reshape(-1)
    if q_target.shape[0] == 0:
        raise ValueError(f"q_target must be non-empty, got shape {q_target.shape}")

    stored_behavior_probability = None
    if record_behavior_probability is not None:
        stored_behavior_probability = float(record_behavior_probability)
        if not np.isfinite(stored_behavior_probability) or stored_behavior_probability <= 0.0:
            raise ValueError(
                f"record_behavior_probability must be finite and strictly positive, got {stored_behavior_probability}"
            )

    return AuxiliaryTrainingRecord(
        context=context,
        skill_id=int(skill_id),
        accept_label=float(bool(accept_label)),
        q_target=tuple(float(v) for v in q_target),
        has_q_target=True,
        behavior_probability=stored_behavior_probability,
        motive_trajectory=None if motive_trajectory is None else tuple(
            tuple(float(value) for value in np.asarray(step, dtype=np.float32).reshape(-1))
            for step in np.asarray(motive_trajectory, dtype=np.float32)
        ),
        candidate_delta_r=all_candidate_delta_r,
        candidate_delta_n=all_candidate_delta_n,
        selected_candidate_index=selected_candidate_index,
    )


def create_auxiliary_trainer_for_model(model: MotiveDecompositionNetwork, seed: int = 0, device: Optional[str] = None) -> MDNAuxiliaryTrainer:
    _seed_everything(seed)
    return MDNAuxiliaryTrainer(model=model, config=MDNAuxiliaryTrainerConfig(random_seed=seed), device=device)
