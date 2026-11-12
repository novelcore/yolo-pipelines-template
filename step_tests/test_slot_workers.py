"""model-training sizes its DataLoader workers by the cores this step may use (kubecore-operator
PRD-HPC-1609): on a shared Slurm node that is its slot, not the whole node os.cpu_count() sees."""
from conftest import import_in


def _cap():
    return import_in("model_training", "app.services.model_training").TrainingService._cap_workers


def test_single_process_loading_stays_single_process(monkeypatch):
    monkeypatch.setenv("KUBECORE_SLOT_CPUS", "32")
    assert _cap()(0) == 0


def test_workers_fit_the_slot_with_one_core_for_the_trainer(monkeypatch):
    monkeypatch.setenv("KUBECORE_SLOT_CPUS", "32")
    assert _cap()(64) == 31
    assert _cap()(8) == 8, "a count that already fits is kept"


def test_a_one_core_slot_still_gets_one_worker(monkeypatch):
    monkeypatch.setenv("KUBECORE_SLOT_CPUS", "1")
    assert _cap()(4) == 1


def test_without_a_slot_the_cpu_binding_counts(monkeypatch):
    import os
    monkeypatch.delenv("KUBECORE_SLOT_CPUS", raising=False)
    bound = len(os.sched_getaffinity(0))
    assert _cap()(10_000) == max(1, bound - 1)
