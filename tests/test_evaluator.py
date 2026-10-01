import json
import os
import re
import sys
import types

import pytest

import lm_eval.api as api
import lm_eval.evaluator as evaluator
from lm_eval import tasks
from lm_eval.api.instance import Instance
from lm_eval.api.filter import FilterEnsemble
from lm_eval.api.metrics import mean
from lm_eval._cli.run import _save_repeat_runs
from lm_eval.filters.selection import MajorityVoteFilter, TakeFirstFilter, TakeKFilter
from lm_eval.loggers import EvaluationTracker
from lm_eval.models.openai_completions import LocalCompletionsAPI
from lm_eval.utils import make_table


os.environ["TOKENIZERS_PARALLELISM"] = "false"
# TODO: more fine grained unit tests rather than this big honking integration
# test once we break evaluator into smaller, more manageable pieces


@pytest.mark.parametrize(
    "task_name,limit,model,model_args,bootstrap_iters",
    [
        (
            ["arc_easy"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            0,
        ),
        (
            ["mmlu_abstract_algebra"],
            None,
            "hf",
            "pretrained=EleutherAI/pythia-160m,dtype=float32,device=cpu",
            10000,
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_evaluator(
    task_name: list[str], limit: int, model: str, model_args: str, bootstrap_iters: int
):
    e1 = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=bootstrap_iters,
    )
    assert e1 is not None

    lm = api.registry.get_model(model).create_from_arg_string(
        model_args,
        {
            "batch_size": None,
            "max_batch_size": None,
            "device": None,
        },
    )
    task_manager = tasks.TaskManager()
    task_dict = task_manager.load(task_name)

    e2 = evaluator.evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
    )

    assert e2 is not None
    # check that caching is working

    def r(x):
        if "arc_easy" in x["results"]:
            return x["results"]["arc_easy"]
        else:
            return x["results"]["mmlu_abstract_algebra"]

    assert all(
        x == y
        for x, y in zip(
            [y for _, y in r(e1).items()], [y for _, y in r(e2).items()], strict=True
        )
    )


@pytest.mark.parametrize(
    "task_name,limit,model,model_args",
    [
        (
            ["ai2_arc"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["mmlu_stem"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["lambada_openai"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
        (
            ["wikitext"],
            10,
            "hf",
            "pretrained=EleutherAI/pythia-14m-deduped,dtype=float32,device=cpu",
        ),
    ],
    ids=lambda d: f"{d}",
)
def test_printed_results(
    task_name: list[str], limit: int, model: str, model_args: str, on_ci: bool
):
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task_name,
        limit=limit,
        model_args=model_args,
        bootstrap_iters=0,
        random_seed=0,
        numpy_random_seed=0,
        torch_random_seed=0,
        fewshot_random_seed=0,
    )

    filename = "_".join(
        (
            "-".join(task_name),
            str(limit),
            str(model),
            re.sub(r"[^a-zA-Z0-9_\-.]", "-", model_args),
        )
    )
    filepath = f"./tests/testdata/{filename}.txt"
    with open(filepath) as f:
        t1 = f.read().strip()

    t2 = make_table(results).strip()

    t1_lines, t2_lines = t1.splitlines(), t2.splitlines()
    assert len(t1_lines) == len(t2_lines)
    for t1_line, t2_line in zip(t1_lines, t2_lines, strict=True):
        t1_items, t2_items = t1_line.split("|"), t2_line.split("|")
        assert len(t1_items) == len(t2_items)
        for t1_item, t2_item in zip(t1_items, t2_items, strict=True):
            try:
                t1_item_f = float(t1_item)
                t2_item_f = float(t2_item)
                ## TODO: these are pretty loose tolerances but:
                # - we only test 10 samples
                # - not sure when/how the ground truth test_data was generated
                tol = 0.3 if on_ci else 0.5
                assert abs(t1_item_f - t2_item_f) < tol
            except ValueError:
                # Strip whitespace so column-width differences
                # (caused by value precision changes) don't fail the test.
                # Also ignore separator-line cells (e.g. "------:").
                t1_s = t1_item.strip().rstrip("-:").rstrip("-")
                t2_s = t2_item.strip().rstrip("-:").rstrip("-")
                if t1_s or t2_s:
                    assert t1_s == t2_s
                #     assert t1_s == t2_s


class _FakeMarkdownTableWriter:
    instances = []

    def __init__(self):
        self.headers = []
        self.value_matrix = []
        type(self).instances.append(self)

    def dumps(self):
        return repr(self.value_matrix)


class _FakeLatexTableWriter(_FakeMarkdownTableWriter):
    pass


@pytest.fixture
def fake_pytablewriter(monkeypatch):
    _FakeMarkdownTableWriter.instances.clear()
    _FakeLatexTableWriter.instances.clear()
    monkeypatch.setitem(
        sys.modules,
        "pytablewriter",
        types.SimpleNamespace(
            MarkdownTableWriter=_FakeMarkdownTableWriter,
            LatexTableWriter=_FakeLatexTableWriter,
        ),
    )


def test_make_table_regression_preserves_hierarchy_and_metadata(fake_pytablewriter):
    result_dict = {
        "results": {
            "parent": {
                "alias": "Parent",
                "name": "ignore-me",
                "sample_len": 123,
                "sample_count": {"acc,none": 123},
                "acc,none": 0.9,
                "acc_stderr,none": 0.01,
            },
            "child": {
                "alias": "Child",
                "sample_len": 50,
                "acc,none": 0.8,
            },
            "standalone": {
                "alias": "Standalone",
                "loss,none": 0.2,
            },
        },
        "versions": {"parent": 1, "child": 2, "standalone": 3},
        "n-shot": {"parent": 5, "child": 0, "standalone": 2},
        "higher_is_better": {
            "parent": {"acc": True},
            "child": {"acc": True},
            "standalone": {"loss": False},
        },
        "group_subtasks": {"parent": ["child"]},
    }

    output = make_table(result_dict)

    assert output == repr(
        [
            ["Parent", 1, "none", "5", "acc", "↑", "0.9000", "±", "0.0100"],
            [" - Child", 2, "none", "0", "acc", "↑", "0.8000", "", ""],
            ["Standalone", 3, "none", "2", "loss", "↓", "0.2000", "", ""],
        ]
    )
    assert _FakeMarkdownTableWriter.instances[0].headers == [
        "Tasks",
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]


def test_make_table_regression_sorted_results_is_alphabetical(fake_pytablewriter):
    result_dict = {
        "results": {
            "parent": {"alias": "Parent", "acc,none": 0.9},
            "child": {"alias": "Child", "acc,none": 0.8},
            "standalone": {"alias": "Standalone", "acc,none": 0.7},
        },
        "versions": {},
        "group_subtasks": {"parent": ["child"]},
    }

    make_table(result_dict, sort_results=True)

    assert _FakeMarkdownTableWriter.instances[0].value_matrix == [
        [" - Child", "    N/A", "none", " ", "acc", "", "0.8000", "", ""],
        ["Parent", "    N/A", "none", " ", "acc", "", "0.9000", "", ""],
        ["Standalone", "    N/A", "none", " ", "acc", "", "0.7000", "", ""],
    ]


def test_local_completions_diagnostic_table():
    class DiagnosticTask:
        task_name = "diagnostic"
        OUTPUT_TYPE = "generate_until"
        VERSION = 3
        eval_docs = [{"answer": "42"} for _ in range(3)]
        _metric_fn_list = {"exact_match": None}

        def __init__(self):
            self.instances = [
                Instance(
                    request_type="generate_until",
                    doc=doc,
                    arguments=(f"Question {idx}", {}),
                    idx=0,
                    metadata=(self.task_name, idx, 1),
                )
                for idx, doc in enumerate(self.eval_docs)
            ]

        def build_all_requests(self, **kwargs):
            pass

        def apply_filters(self):
            for instance in self.instances:
                response = instance.resps[0]
                instance.filtered_resps = {
                    "strict-match": (
                        "42" if "The answer is 42" in response else "[invalid]"
                    ),
                    "flexible-extract": "42" if "42" in response else "[invalid]",
                }

        def doc_iterator(self, **kwargs):
            return enumerate(self.eval_docs)

        def process_results(self, doc, responses):
            return {"exact_match": float(responses[0] == doc["answer"])}

        def doc_to_target(self, doc):
            return doc["answer"]

        def aggregation(self):
            return {"exact_match": mean}

        def higher_is_better(self):
            return {"exact_match": True}

        def dump_config(self):
            return {"task": self.task_name, "num_fewshot": 0}

    class DiagnosticLM:
        rank = 0
        world_size = 1

        def generate_until(self, requests):
            parser = LocalCompletionsAPI.__new__(LocalCompletionsAPI)
            parser.think_end_token = "</mm:think>"
            raw_texts = [
                "<mm:think>work</mm:think>The answer is 42.",
                "<mm:think>still working 42",
                "<mm:think>work</mm:think>42",
            ]
            return [
                parser.parse_generations(
                    {"choices": [{"index": 0, "text": raw_texts[req.doc_id]}]}
                )[0]
                for req in requests
            ]

    result = evaluator.evaluate(
        lm=DiagnosticLM(),
        task_dict={"tasks": {"diagnostic": DiagnosticTask()}, "groups": {}},
        bootstrap_iters=0,
        log_samples=True,
    )

    stats = result["diagnostic_stats"]["diagnostic"]
    assert stats["answer_not_found"] == 1
    assert stats["invalid_filter"] == {
        "strict-match": 1,
        "flexible-extract": 0,
    }
    assert len(result["diagnostic_samples"]["diagnostic"]["not_found"]) == 1
    assert len(result["diagnostic_samples"]["diagnostic"]["invalid"]["strict-match"]) == 1

    table = make_table(result)
    assert "answer-not-found|invalid-filter" in table
    rows = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in table.splitlines()[2:]
    ]
    ratios_by_filter = {row[2]: row[-2:] for row in rows}
    assert ratios_by_filter == {
        "strict-match": ["0.3333", "0.3333"],
        "flexible-extract": ["0.3333", "0.0000"],
    }


@pytest.mark.parametrize("report_repeat_stats", [False, True])
def test_repeat_statistics_are_opt_in_and_preserve_existing_scores(
    report_repeat_stats, caplog, monkeypatch, tmp_path
):
    monkeypatch.setenv(
        "LMEVAL_LOG_LEVEL", "DEBUG" if report_repeat_stats else "INFO"
    )
    class RepeatTask:
        task_name = "repeat_task"
        OUTPUT_TYPE = "generate_until"
        VERSION = 1
        eval_docs = [{"answer": "yes"}, {"answer": "no"}]
        _metric_fn_list = {"accuracy": None}

        def __init__(self):
            self._config = types.SimpleNamespace(
                report_repeat_stats=report_repeat_stats, repeats=3
            )
            self._filters = [
                FilterEnsemble("first", [TakeFirstFilter]),
                FilterEnsemble("majority", [MajorityVoteFilter, TakeFirstFilter]),
                FilterEnsemble("first-two", [lambda: TakeKFilter(k=2), MajorityVoteFilter, TakeFirstFilter]),
            ]
            self.instances = [
                Instance("generate_until", doc, ("prompt", {}), 0, (self.task_name, idx, 1))
                for idx, doc in enumerate(self.eval_docs)
            ]

        def build_all_requests(self, **kwargs):
            pass

        def apply_filters(self):
            for ensemble in self._filters:
                ensemble.apply(self.instances)

        def doc_iterator(self, **kwargs):
            return enumerate(self.eval_docs)

        def process_results(self, doc, responses):
            return {"accuracy": float(responses[0] == doc["answer"])}

        def doc_to_target(self, doc):
            return doc["answer"]

        def aggregation(self):
            return {"accuracy": mean}

        def higher_is_better(self):
            return {"accuracy": True}

        def dump_config(self):
            return {"task": self.task_name, "report_repeat_stats": report_repeat_stats}

        def get_config(self, key):
            return getattr(self._config, key, None)

    class RepeatLM:
        rank = 0
        world_size = 1

        def generate_until(self, requests):
            return ["yes", "", "yes", "no", "no", "[invalid]"]

    with caplog.at_level("WARNING"):
        result = evaluator.evaluate(
            lm=RepeatLM(),
            task_dict={"tasks": {"repeat_task": RepeatTask()}, "groups": {}},
            bootstrap_iters=100 if report_repeat_stats else 0,
            log_samples=report_repeat_stats,
        )
    scores = result["results"]["repeat_task"]
    assert scores["accuracy,first"] == 1.0
    assert scores["accuracy,majority"] == 1.0
    assert scores["accuracy,first-two"] == 1.0
    if report_repeat_stats:
        assert scores["accuracy_min_repeats,first"] == 0.5
        assert scores["accuracy_mean_repeats,first"] == pytest.approx(2 / 3)
        assert scores["accuracy_median_repeats,first"] == 0.5
        assert scores["accuracy_max_repeats,first"] == 1.0
        assert "accuracy_mean_repeats,first-two" not in scores
        assert "Skipping repeat statistics for filter first-two" in caplog.text
        repeat_results = result["repeat_results"]
        assert len(repeat_results) == 3
        assert repeat_results[0]["results"]["repeat_task"]["accuracy,first"] == 1.0
        assert repeat_results[1]["results"]["repeat_task"]["accuracy,first"] == 0.5
        assert isinstance(
            repeat_results[1]["results"]["repeat_task"]["accuracy_stderr,first"],
            float,
        )
        assert repeat_results[1]["diagnostic_stats"]["repeat_task"]["answer_not_found"] == 1
        assert repeat_results[2]["diagnostic_stats"]["repeat_task"]["invalid_filter"]["first"] == 1
        table = make_table(repeat_results[1])
        assert "answer-not-found|invalid-filter" in table
        assert "0.5000" in table
        repeat_samples = result["repeat_samples"]["repeat_task"]
        assert len(repeat_samples) == 3
        assert len(repeat_samples[0]) == 4  # two docs, two compatible filters
        assert all(sample["resps"] == [["yes"]] for sample in repeat_samples[0] if sample["doc_id"] == 0)
        assert all(sample["resps"] == [[""]] for sample in repeat_samples[1] if sample["doc_id"] == 0)
        assert all(sample["accuracy"] == 0.0 for sample in repeat_samples[1] if sample["doc_id"] == 0)

        tracker = EvaluationTracker(output_path=str(tmp_path))
        tracker.general_config_tracker.log_experiment_args(
            model_source="test",
            model_args={"model": "test_model"},
            system_instruction=None,
            chat_template=None,
            fewshot_as_multiturn=False,
        )
        base_result = {
            key: value
            for key, value in result.items()
            if key not in ("samples", "repeat_results", "repeat_samples")
        }
        tracker.save_results_aggregated(results=base_result, samples=result["samples"])
        tracker.save_results_samples(
            task_name="repeat_task", samples=result["samples"]["repeat_task"]
        )
        _save_repeat_runs(tracker, base_result, repeat_results, result["repeat_samples"])

        repeat_dirs = sorted(tmp_path.glob("*_repeat*"))
        assert len(repeat_dirs) == 3
        assert (tmp_path / "test_model").is_dir()  # combined run remains
        second_model_dir = repeat_dirs[1] / "test_model"
        result_files = list(second_model_dir.glob("results_*.json"))
        files = list(second_model_dir.glob("samples_repeat_task_*.jsonl"))
        assert len(result_files) == len(files) == 1
        second_result = json.loads(result_files[0].read_text())
        assert second_result["repeat_index"] == 2
        assert second_result["results"]["repeat_task"]["accuracy,first"] == 0.5
        second_repeat = [json.loads(line) for line in files[0].read_text().splitlines()]
        assert len(second_repeat) == 4
        assert all(sample["resps"] == [[""]] for sample in second_repeat if sample["doc_id"] == 0)
        not_found_file = next(second_model_dir.glob("samples_not_found_repeat_task_*.jsonl"))
        not_found_sample = json.loads(not_found_file.read_text().splitlines()[0])
        assert not_found_sample["arguments"]["gen_args_0"]["arg_0"] == "prompt"
        assert repeat_samples[1][0]["arguments"] == [("prompt", {})]
        assert list((repeat_dirs[2] / "test_model").glob("samples_invalid_repeat_task_first_*.jsonl"))
    else:
        assert "accuracy_mean_repeats,first" not in scores
        assert "repeat_results" not in result
