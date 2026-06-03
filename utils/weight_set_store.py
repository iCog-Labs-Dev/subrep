"""Persistent monotone weight-set store for context-conditioned W_x tracking."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import json

import numpy as np

from utils.support_geometry import compute_support_values_from_vertices, make_basis_query_directions, simplex_support_values


@dataclass
class WeightSet:
    """Weight set W_x for a single context, represented by observed vertices."""

    vertices: list[np.ndarray] = field(default_factory=list)

    def is_empty(self) -> bool:
        return len(self.vertices) == 0

    def add_vertex(self, weight_vector: np.ndarray) -> None:
        weight_vector = np.asarray(weight_vector, dtype=np.float32).reshape(-1)
        if weight_vector.ndim != 1 or len(weight_vector) == 0:
            raise ValueError(f"weight_vector must be a non-empty 1D vector, got {weight_vector.shape}")
        if not np.all(np.isfinite(weight_vector)):
            raise ValueError("weight_vector must contain only finite values")
        self.vertices.append(weight_vector.copy())

    def get_support_values(self, query_directions: np.ndarray) -> np.ndarray:
        if self.is_empty():
            return simplex_support_values(query_directions)
        vertices_array = np.stack(self.vertices, axis=0)
        return compute_support_values_from_vertices(vertices_array, query_directions)

    def get_vertices_array(self) -> Optional[np.ndarray]:
        if self.is_empty():
            return None
        return np.stack(self.vertices, axis=0)


class WeightSetStore:
    """Per-context registry of learned weight sets W_x."""

    def __init__(self, num_objectives: int) -> None:
        if num_objectives <= 0:
            raise ValueError(f"num_objectives must be positive, got {num_objectives}")
        self.num_objectives = int(num_objectives)
        self._store: dict[tuple[float, ...], WeightSet] = {}
        self._query_directions = make_basis_query_directions(num_objectives)

    def _context_key(self, context: np.ndarray) -> tuple[float, ...]:
        context = np.asarray(context, dtype=np.float32).reshape(-1)
        if context.ndim != 1 or len(context) == 0:
            raise ValueError(f"context must be a non-empty 1D vector, got {context.shape}")
        if not np.all(np.isfinite(context)):
            raise ValueError("context must contain only finite values")
        return tuple(np.round(context, decimals=4).tolist())

    def observe_certified_weight(self, context: np.ndarray, weight_vector: np.ndarray) -> None:
        key = self._context_key(context)
        if key not in self._store:
            self._store[key] = WeightSet()
        self._store[key].add_vertex(weight_vector)

    def get_support_values(self, context: np.ndarray) -> np.ndarray:
        key = self._context_key(context)
        weight_set = self._store.get(key, WeightSet())
        return weight_set.get_support_values(self._query_directions)

    def get_weight_set(self, context: np.ndarray) -> Optional[WeightSet]:
        """Get the WeightSet for a context, or None if not yet observed."""
        key = self._context_key(context)
        return self._store.get(key)

    def get_all_support_targets(self) -> list[tuple[np.ndarray, np.ndarray]]:
        targets: list[tuple[np.ndarray, np.ndarray]] = []
        for key, weight_set in self._store.items():
            context = np.array(key, dtype=np.float32)
            support_values = weight_set.get_support_values(self._query_directions)
            targets.append((context, support_values))
        return targets

    def context_count(self) -> int:
        return len(self._store)

    def total_vertex_count(self) -> int:
        return sum(len(weight_set.vertices) for weight_set in self._store.values())

    def save(self, path: str | Path) -> None:
        """Persist the current `W_x` store to JSON."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "num_objectives": self.num_objectives,
            "contexts": {
                ",".join(map(str, key)): [vertex.tolist() for vertex in weight_set.vertices]
                for key, weight_set in self._store.items()
            },
        }
        file_path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "WeightSetStore":
        """Restore a `WeightSetStore` from JSON persistence."""
        file_path = Path(path)
        data = json.loads(file_path.read_text(encoding="utf-8"))
        store = cls(num_objectives=int(data["num_objectives"]))
        for key_str, vertices_list in data["contexts"].items():
            key = tuple(float(value) for value in key_str.split(",") if value != "")
            weight_set = WeightSet()
            for vertex in vertices_list:
                weight_set.add_vertex(np.asarray(vertex, dtype=np.float32))
            store._store[key] = weight_set
        return store
