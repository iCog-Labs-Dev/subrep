"""Train/evaluate the policy-conditioned generator exclusively from raw_mixed."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from generator.skill_generator import SkillGenerator, PolicyConditionedSkillGenerator
from generator.multi_policy_data import (
    load_records, load_manifest, load_or_create_manifest, partition,
)
from generator.policy_registry import POLICY_NAMES


def dataset(rows):
    return TensorDataset(
        torch.tensor(np.stack([r["obs"] for r in rows]), dtype=torch.float32),
        torch.tensor([r["policy_id"] for r in rows], dtype=torch.long),
        torch.tensor([[r["payoff"]] for r in rows], dtype=torch.float32),
        torch.tensor(np.stack([r["motives"] for r in rows]), dtype=torch.float32),
    )


def fit(
    rows,
    validation,
    *,
    conditioned=True,
    epochs=50,
    patience=10,
    lr=1e-3,
    batch_size=32,
    hidden_dim=64,
    embedding_dim=4,
    seed=42,
):
    torch.manual_seed(seed)
    model = (
        PolicyConditionedSkillGenerator(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            num_policies=len(POLICY_NAMES),
        )
        if conditioned
        else SkillGenerator(hidden_dim=hidden_dim)
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = DataLoader(dataset(rows), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset(validation), batch_size=batch_size)

    best, best_state, wait = float("inf"), None, 0
    print("\nStarting training...")

    for epoch in range(epochs):
        model.train()
        training_losses = []

        for obs, ids, payoff, motives in train_loader:
            optimizer.zero_grad()
            pred_p, pred_m = model(obs, ids) if conditioned else model(obs)
            loss = (
                torch.nn.functional.mse_loss(pred_p, payoff)
                + torch.nn.functional.mse_loss(pred_m, motives)
            )
            loss.backward()
            optimizer.step()
            training_losses.append((len(obs), float(loss.item())))

        model.eval()
        with torch.no_grad():
            losses = []
            for obs, ids, payoff, motives in val_loader:
                p, m = model(obs, ids) if conditioned else model(obs)
                losses.append(
                    (
                        len(obs),
                        float(
                            torch.nn.functional.mse_loss(p, payoff)
                            + torch.nn.functional.mse_loss(m, motives)
                        ),
                    )
                )
            val_loss = sum(n * loss for n, loss in losses) / sum(
                n for n, _ in losses
            )

        train_loss = sum(n * loss for n, loss in training_losses) / sum(
            n for n, _ in training_losses
        )

        if (epoch + 1) % 5 == 0 or epoch == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}",
                flush=True,
            )

        if val_loss < best:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(
                    f"\nEarly stopping at epoch {epoch + 1}: "
                    f"no val_loss improvement for {patience} epochs.",
                    flush=True,
                )
                break

    model.load_state_dict(best_state)
    return model, dict(best_validation_loss=best, epochs_run=epoch + 1)


def _manifest_provenance(manifest):
    """Return the dataset/split identity required in every multi-policy checkpoint."""
    required = ("dataset_signature", "split_signature")
    missing = [key for key in required if not manifest.get(key)]
    if missing:
        raise ValueError(f"Cannot save/load checkpoint: manifest lacks {missing}")
    return {key: manifest[key] for key in required}


def save_model(
    model,
    path,
    *,
    conditioned,
    manifest,
    hidden_dim=64,
    embedding_dim=4,
):
    """Save model weights together with the exact dataset and split identities."""
    provenance = _manifest_provenance(manifest)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        dict(
            state_dict=model.state_dict(),
            conditioned=conditioned,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            policy_names=list(POLICY_NAMES),
            **provenance,
        ),
        p,
    )


def load_model(path, *, expected_manifest=None):
    """Load a checkpoint and optionally verify it against the evaluation manifest."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)

    if checkpoint.get("policy_names") != list(POLICY_NAMES):
        raise ValueError("Checkpoint policy registry mismatch")

    checkpoint_provenance = {
        key: checkpoint.get(key)
        for key in ("dataset_signature", "split_signature")
    }

    if expected_manifest is not None:
        if any(value is None for value in checkpoint_provenance.values()):
            raise ValueError(
                "Checkpoint is missing dataset/split provenance. "
                "Retrain the model using the current multi-policy pipeline before evaluation."
            )

        expected = _manifest_provenance(expected_manifest)
        if checkpoint_provenance != expected:
            raise ValueError(
                "Checkpoint dataset/split identity does not match the saved training "
                "manifest. Refusing to evaluate on a different split."
            )

    model = (
        PolicyConditionedSkillGenerator(
            hidden_dim=checkpoint["hidden_dim"],
            embedding_dim=checkpoint["embedding_dim"],
        )
        if checkpoint["conditioned"]
        else SkillGenerator(hidden_dim=checkpoint["hidden_dim"])
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint["conditioned"]


def predict(model, rows, conditioned):
    model.eval()
    with torch.no_grad():
        obs, ids, _, _ = dataset(rows).tensors
        p, m = model(obs, ids) if conditioned else model(obs)
    return np.column_stack([p.numpy().reshape(-1), m.numpy()])


def metric_report(targets, predictions, rows):
    def metrics(indices):
        diff = predictions[indices] - targets[indices]
        return {
            f"{name}_{kind}": float(
                np.mean(np.abs(diff[:, j]))
                if kind == "mae"
                else np.mean(diff[:, j] ** 2)
            )
            for j, name in enumerate(("payoff", "safety", "fuel"))
            for kind in ("mae", "mse")
        }

    return dict(
        overall=metrics(np.arange(len(rows))),
        per_policy={
            name: metrics(
                np.array([i for i, row in enumerate(rows) if row["policy"] == name])
            )
            for name in POLICY_NAMES
        },
    )


def evaluate(rows, train_rows, models):
    target = np.array([[r["payoff"], *r["motives"]] for r in rows])
    means = {
        name: np.mean(
            [
                [r["payoff"], *r["motives"]]
                for r in train_rows
                if r["policy"] == name
            ],
            axis=0,
        )
        for name in POLICY_NAMES
    }

    output = {
        "per_policy_training_mean": metric_report(
            target,
            np.stack([means[row["policy"]] for row in rows]),
            rows,
        )
    }

    for name, (model, conditioned) in models.items():
        output[name] = metric_report(
            target, predict(model, rows, conditioned), rows
        )
    return output


def prepare(args):
    """Training may create a manifest; evaluation commands must reuse one."""
    rows = load_records(args.data_dir)

    if args.command == "train":
        manifest = load_or_create_manifest(
            rows,
            args.split_manifest,
            seed=args.seed,
        )
    else:
        manifest = load_manifest(rows, args.split_manifest)

    return partition(rows, manifest), manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("train", "evaluate", "sizes"))
    parser.add_argument("--data-dir", default="data/raw_mixed")
    parser.add_argument("--split-manifest", default="data/mixed_split_manifest.json")
    parser.add_argument("--model-path", default="models/multi_policy_generator.pt")
    parser.add_argument("--baseline-path", default="models/observation_only_baseline.pt")
    parser.add_argument("--output-dir", default="reports/multi_policy")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--embedding-dim", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", nargs="+", type=int, default=[25, 50, 100])
    args = parser.parse_args(argv)

    parts, manifest = prepare(args)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.command == "train":
        for conditioned, path in (
            (True, args.model_path),
            (False, args.baseline_path),
        ):
            print(
                f"\nTraining {'multi-policy' if conditioned else 'observation-only'} "
                "generator...",
                flush=True,
            )
            model, stats = fit(
                parts["train"],
                parts["val"],
                conditioned=conditioned,
                epochs=args.epochs,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                seed=args.seed,
            )
            save_model(
                model,
                path,
                conditioned=conditioned,
                manifest=manifest,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
            )
            print(
                f"Best model (val_loss={stats['best_validation_loss']:.6f}) "
                f"saved -> {path}",
                flush=True,
            )

    elif args.command == "evaluate":
        models = {
            "multi_policy": load_model(
                args.model_path,
                expected_manifest=manifest,
            ),
            "observation_only": load_model(
                args.baseline_path,
                expected_manifest=manifest,
            ),
        }
        result = evaluate(parts["test"], parts["train"], models)
        (out / "prediction_metrics.json").write_text(
            json.dumps(result, indent=2)
        )
        print(json.dumps({key: value["overall"] for key, value in result.items()}, indent=2))
        print(f"  Saved -> {out / 'prediction_metrics.json'}")

    else:
        contexts = sorted({row["context_id"] for row in parts["train"]})
        import random

        random.Random(args.seed).shuffle(contexts)
        results = []

        for n in args.sizes:
            print(f"\nTraining subset target: {n} contexts", flush=True)
            if n < 1 or n > len(contexts):
                raise ValueError(
                    f"Size {n} outside 1..{len(contexts)} training contexts"
                )

            subset = [
                row for row in parts["train"]
                if row["context_id"] in set(contexts[:n])
            ]
            model, stats = fit(
                subset,
                parts["val"],
                epochs=args.epochs,
                patience=args.patience,
                hidden_dim=args.hidden_dim,
                embedding_dim=args.embedding_dim,
                seed=args.seed,
            )
            result = evaluate(
                parts["test"],
                subset,
                {"multi_policy": (model, True)},
            )
            results.append(
                dict(
                    contexts=n,
                    records=len(subset),
                    validation=stats,
                    test_metrics=result,
                )
            )

        (out / "dataset_size_metrics.json").write_text(
            json.dumps(results, indent=2)
        )
        print(f'Saved {out / "dataset_size_metrics.json"}')


if __name__ == "__main__":
    main()