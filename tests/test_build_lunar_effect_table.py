"""How the effect table builder shares its measurements out between processes."""

import torch

from demo import build_lunar_effect_table as builder


class _RecordingPool:
    """Stands in for ProcessPoolExecutor and records how its processes would start."""

    started_with = None

    def __init__(self, max_workers, mp_context):
        _RecordingPool.started_with = mp_context.get_start_method()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def map(self, fn, jobs):
        return [[] for _ in jobs]


def test_measuring_processes_start_fresh_on_every_system(monkeypatch):
    monkeypatch.setattr(builder, "ProcessPoolExecutor", _RecordingPool)

    assert builder.measure(range(3), workers=2) == []
    assert _RecordingPool.started_with == "spawn"


def test_a_seed_measured_in_a_fresh_process_is_measured_as_in_this_one():
    threads = torch.get_num_threads()
    try:
        in_process = builder._measure(([3], None))  # what each process runs; it sets torch to one thread
    finally:
        torch.set_num_threads(threads)

    assert builder.measure(range(3, 4), workers=1) == in_process  # seed 3 is among the quickest to measure
    assert in_process[0].flights
