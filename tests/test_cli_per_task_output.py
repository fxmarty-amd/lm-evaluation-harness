"""Per task / group output layout of `lm_eval run` (results, samples, repeats)."""

import json

from lm_eval._cli.run import _results_subset, _save_per_task, _task_units
from lm_eval.loggers import EvaluationTracker


def _sample(doc_id: int, resp: str = "x") -> dict:
    return {
        "doc_id": doc_id,
        "arguments": [("prompt", {})],
        "resps": [[resp]],
        "filtered_resps": [resp],
        "target": "y",
        "doc_hash": f"d{doc_id}",
        "prompt_hash": "p",
        "target_hash": "t",
    }


def _results() -> dict:
    """mmlu_pro_chat group (2 subjects), standalone aime26, and a nested group."""
    tasks = ["mmlu_pro_chat_biology", "mmlu_pro_chat_law", "aime26", "leaf"]
    groups = ["mmlu_pro_chat", "outer", "inner"]
    return {
        "results": {t: {"exact_match,none": 0.5} for t in tasks},
        "groups": {g: {"exact_match,none": 0.5} for g in groups},
        "group_subtasks": {
            "mmlu_pro_chat": ["mmlu_pro_chat_biology", "mmlu_pro_chat_law"],
            "outer": ["inner"],
            "inner": ["leaf"],
        },
        "configs": {t: {"task": t, "repeats": 2 if t == "aime26" else 1} for t in tasks},
        "versions": {t: 1 for t in tasks + groups},
        "n-samples": {t: {"original": 2, "effective": 2} for t in tasks},
        "config": {"model": "test", "batch_size": 1},
        "date": 0.0,
    }


def _tracker(output_path) -> EvaluationTracker:
    tracker = EvaluationTracker(output_path=str(output_path))
    tracker.general_config_tracker.log_experiment_args(
        model_source="test",
        model_args={"model": "test_model"},
        system_instruction=None,
        chat_template=None,
        fewshot_as_multiturn=False,
    )
    return tracker


def _save(tmp_path, output_path=None):
    results = _results()
    samples = {t: [_sample(0), _sample(1)] for t in results["configs"]}
    diagnostic = {
        "aime26": {"not_found": [_sample(1)], "invalid": {"none": [_sample(0)]}},
        "mmlu_pro_chat_law": {"not_found": [], "invalid": {}},
    }
    repeat_results = [
        {"results": {"aime26": {"exact_match,none": 1.0}}, "repeat_index": 1},
        {"results": {"aime26": {"exact_match,none": 0.0}}, "repeat_index": 2},
    ]
    repeat_samples = {"aime26": [[_sample(0, "r0")], [_sample(0, "r1")]]}
    tracker = _tracker(output_path or tmp_path)
    tracker.save_results_aggregated(results=results, samples=samples)
    _save_per_task(tracker, results, samples, diagnostic, repeat_results, repeat_samples)
    return tracker


def _one(directory, pattern):
    files = list(directory.glob(pattern))
    assert len(files) == 1, (directory, pattern, files)
    return files[0]


def test_task_units_maps_top_level_tasks_and_groups():
    units = _task_units(_results())
    assert units == {
        "mmlu_pro_chat": ["mmlu_pro_chat", "mmlu_pro_chat_biology", "mmlu_pro_chat_law"],
        "outer": ["outer", "inner", "leaf"],
        "aime26": ["aime26"],
    }


def test_results_subset_filters_task_keyed_sections_only():
    results = _results()
    known = {n for names in _task_units(results).values() for n in names}
    subset = _results_subset(results, ["aime26"], known)
    assert list(subset["results"]) == ["aime26"]
    assert subset["groups"] == {}
    assert subset["group_subtasks"] == {}
    assert list(subset["configs"]) == list(subset["versions"]) == ["aime26"]
    assert subset["config"] == results["config"]  # not task-keyed: kept as is
    assert subset["date"] == results["date"]


def test_per_task_layout(tmp_path):
    _save(tmp_path)
    model_dir = tmp_path / "test_model"
    combined = json.loads(_one(model_dir, "results_*.json").read_text())
    assert set(combined["results"]) == {
        "mmlu_pro_chat_biology",
        "mmlu_pro_chat_law",
        "aime26",
        "leaf",
    }
    # No sample files next to the combined results, no top-level repeat folders.
    assert not list(model_dir.glob("samples_*.jsonl"))
    assert not list(tmp_path.glob("*_repeat*"))
    assert sorted(p.name for p in model_dir.iterdir() if p.is_dir()) == [
        "aime26",
        "mmlu_pro_chat",
        "outer",
    ]

    mmlu = model_dir / "mmlu_pro_chat"
    mmlu_results = json.loads(_one(mmlu, "results_*.json").read_text())
    assert set(mmlu_results["results"]) == {"mmlu_pro_chat_biology", "mmlu_pro_chat_law"}
    assert set(mmlu_results["groups"]) == {"mmlu_pro_chat"}
    assert set(mmlu_results["task_hashes"]) == {"mmlu_pro_chat_biology", "mmlu_pro_chat_law"}
    _one(mmlu, "samples_mmlu_pro_chat_biology_*.jsonl")
    _one(mmlu, "samples_mmlu_pro_chat_law_*.jsonl")
    assert not list(mmlu.glob("repeat*"))  # single repeat
    assert not list(mmlu.glob("samples_not_found_*"))  # nothing to report

    outer = model_dir / "outer"
    assert set(json.loads(_one(outer, "results_*.json").read_text())["groups"]) == {
        "outer",
        "inner",
    }
    _one(outer, "samples_leaf_*.jsonl")

    aime = model_dir / "aime26"
    assert set(json.loads(_one(aime, "results_*.json").read_text())["results"]) == {
        "aime26"
    }
    assert len(_one(aime, "samples_aime26_*.jsonl").read_text().splitlines()) == 2
    _one(aime, "samples_not_found_aime26_*.jsonl")
    _one(aime, "samples_invalid_aime26_none_*.jsonl")
    assert sorted(p.name for p in aime.glob("repeat*")) == ["repeat0", "repeat1"]
    for k, (resp, score) in enumerate([("r0", 1.0), ("r1", 0.0)]):
        repeat_dir = aime / f"repeat{k}"
        repeat_results = json.loads(_one(repeat_dir, "results_*.json").read_text())
        assert repeat_results["repeat_index"] == k + 1
        assert repeat_results["results"] == {"aime26": {"exact_match,none": score}}
        assert "groups" not in repeat_results
        line = _one(repeat_dir, "samples_aime26_*.jsonl").read_text().splitlines()
        assert json.loads(line[0])["resps"] == [[resp]]


def test_per_task_layout_with_json_output_path(tmp_path):
    _save(tmp_path, output_path=tmp_path / "run.json")
    _one(tmp_path, "run_*.json")  # combined results, as before
    _one(tmp_path / "aime26", "results_*.json")
    _one(tmp_path / "aime26" / "repeat1", "samples_aime26_*.jsonl")
    _one(tmp_path / "mmlu_pro_chat", "samples_mmlu_pro_chat_law_*.jsonl")


def test_per_task_layout_without_samples(tmp_path):
    # --log_samples off: per task results and repeat results only.
    results = _results()
    tracker = _tracker(tmp_path)
    tracker.save_results_aggregated(results=results)
    repeat_results = [{"results": {"aime26": {"exact_match,none": 1.0}}}]
    _save_per_task(tracker, results, None, {}, repeat_results, {})
    model_dir = tmp_path / "test_model"
    assert not list(model_dir.rglob("samples_*.jsonl"))
    _one(model_dir / "aime26", "results_*.json")
    _one(model_dir / "aime26" / "repeat0", "results_*.json")
    _one(model_dir / "mmlu_pro_chat", "results_*.json")


def test_save_per_task_without_output_path_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    tracker = EvaluationTracker(output_path=None)
    _save_per_task(tracker, _results(), None, {}, [], {})
    assert not list(tmp_path.iterdir())

