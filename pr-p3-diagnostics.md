<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

This brings more transparency to users as to why they are getting a certain evaluation score for generative tasks, which may be different from the numbers reported in papers.

Fixes https://github.com/EleutherAI/lm-evaluation-harness/issues/3382

Likely fixes: https://github.com/EleutherAI/lm-evaluation-harness/issues/3003 https://github.com/EleutherAI/lm-evaluation-harness/issues/1292 https://github.com/EleutherAI/lm-evaluation-harness/issues/2847 https://github.com/EleutherAI/lm-evaluation-harness/issues/2707 https://github.com/EleutherAI/lm-evaluation-harness/issues/3129 https://github.com/EleutherAI/lm-evaluation-harness/issues/3048 https://github.com/EleutherAI/lm-evaluation-harness/issues/2941 https://github.com/EleutherAI/lm-evaluation-harness/issues/2953

Namely:

- Adds `answer-not-found` and `invalid-filter` diagnostic columns to the evaluation summary table, visible whenever the run has results (shown as `N/A` for the tasks they do not apply to, i.e. non-generative ones and the backends that do not report the metadata yet).
  - `answer-not-found`: when the configured `think_end_token` is missing from the model output, or when `max_gen_toks` is hit.
  - `invalid-filter`: when regex extraction falls back to `[invalid]`.
- Logs the diagnostic samples (full responses, stop reasons) to separate JSONL files when `--log_samples` is enabled.

This replaces #3709, rebased on current `main` and split away from unrelated repeat-statistics work.

## Claims

- `results["diagnostic_stats"][<task>]` reports `total`, `answer_not_found`, `filter_total` and `invalid_filter` per filter, summed across ranks and aggregated over leaf tasks for task groups.
- The results table gains `answer-not-found` and `invalid-filter` columns as ratios of the per-filter totals, so a run that produced no answer at all is immediately visible:

```
| Tasks  | Version | Filter | n-shot | Metric      | Value  | Stderr | answer-not-found | invalid-filter |
|:-------|--------:|:-------|-------:|:------------|-------:|-------:|-----------------:|---------------:|
| aime26 |       0 | none   |      0 | exact_match | 0.8667 | 0.0631 |           0.1333 |         0.0000 |
```

- With `--log_samples`, the offending samples are written next to the ordinary ones as `samples_not_found_<task>_<ts>.jsonl` and `samples_invalid_<task>_<filter>_<ts>.jsonl`, carrying the full response, the stop reason and the two flags:

```json
{"doc_id": 10, "target": "896", "filter": "none",
 "resps": [["[THINK]Let me reconsider the boundary terms…"]],
 "full_resps": [["[THINK]Let me reconsider the boundary terms…"]],
 "stop_reasons": [["max_gen_toks"]], "found_answers": [[false]],
 "metrics": ["exact_match"], "exact_match": 0.0,
 "answer_not_found": true, "invalid_filter": false}
```

- A response is "missing" when it is empty, `None`, has `answer_found is False`, or stopped at the token budget (`stop_reason` of `max_gen_toks`, or the OpenAI/vLLM spelling `length`). That last case is what makes the column useful for models **without** a reasoning trace too, where reaching the limit is the same symptom: either no answer, or the stop condition was wrong and the model kept going after answering. A response is "invalid" only when it is *not* missing, so the two counters are disjoint and a failed parse is never double-counted. Missing-answer samples are collected once per document (the first filter), since every filter sees the same generation; invalid ones are collected per filter, since filters fail differently.

Left to other PRs:
- [ ] The native backends (`hf`, vLLM, SGLang) do not log `full_resps` / `stop_reasons` / `found_answers` yet, so this PR only supports the OpenAI-compatible clients (`local-completions`, `local-chat-completions`, `openai-*`); the `answer_found is False` clause and the per-sample stop reasons rely on that metadata, and the native backends will report `N/A` until it is standardized.
- [ ] Support non-single token `think_end_token` (e.g. gpt-oss that needs `<|channel|>final<|message|>`), setting it by default without the user having to specify it.
- [ ] `[missing]` sentinel in `RegexFilter` and `MultiChoiceRegexFilter` for empty responses: the counting here does not need it, so it is not part of this change.
- [ ] Per-repeat diagnostics; that belongs with the repeat-statistics PR.

## Validation

### Unit tests

```
$ python -m pytest tests/test_evaluator.py -q -k "diagnostic or answer_not_found"
3 passed, 8 deselected in 0.17s

$ python -m pytest tests/test_evaluator.py -q -k "printed_results"
4 passed, 7 deselected in 103.12s

$ python -m pytest tests/models/test_api.py -q
15 passed in 5.42s
```

The `printed_results` reference tables in `tests/testdata` were updated for the two new columns (all four samples are non-generative, so they read `N/A`).

The tests assert the counts (`answer_not_found`, `invalid_filter` per filter), the collected samples, the exact rendered table including the new columns, and the missing-answer rule itself: truncated with `max_gen_toks` and with `length`, clean stops (`natural`, `stop`, `stop_sequence:<x>`) not counted, empty response, unclosed trace, and `loglikelihood` requests ignored.

### End-to-end

The example this replaces, showing why the columns matter: `gsm8k_platinum` with `max_gen_toks` set too small to hold a reasoning trace. Every answer is "wrong" for a reason the score alone cannot express.

```bash
export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
export PRETRAINED_PATH="Qwen/Qwen3-8B"
export TASKS="gsm8k_platinum,mmlu_redux_high_school_mathematics_generative"
export LOG_PATH="/tmp/results/${PRETRAINED_PATH}/${TIMESTAMP}_lm_eval"

mkdir -p ${LOG_PATH}
lm_eval \
  --model vllm \
  --model_args '{"pretrained":"'"${PRETRAINED_PATH}"'","dtype":"auto","tensor_parallel_size":1,"enable_thinking": true,"think_end_token":"</think>"}' \
  --device "cuda" \
  --tasks ${TASKS} \
  --apply_chat_template \
  --fewshot_as_multiturn \
  --log_samples \
  --output_path ${LOG_PATH} \
  --batch_size "auto" &> ${LOG_PATH}/lm_eval.log
```

gives:

```
|         Tasks         |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|answer-not-found|invalid-filter|
|-----------------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|---------------:|-------------:|
|gsm8k_platinum         |      3|flexible-extract|     5|exact_match|↑  |0.0091|±  |0.0027|          0.9909|             0|
|                       |       |strict-match    |     5|exact_match|↑  |0.0091|±  |0.0027|          0.9909|             0|
|high_school_mathematics|      4|default         |     0|exact_match|↑  |0.1000|±  |0.0302|          0.8200|             0|
```

with e.g. `…/Qwen__Qwen3-8B/samples_not_found_gsm8k_platinum_<ts>.jsonl` holding the documents that produced no answer, together with their `stop_reasons` and `full_resps`, i.e. exactly the evidence needed to conclude that the generation budget, not the model, is the problem.

Cross-checked on an OpenAI-compatible endpoint (8× MI350, ROCm 7.0): 30 reasoning problems, 100k-token budget, greedy. The run scored 0.8667 with `answer-not-found = 0.1333`, and `samples_not_found_aime26_*.jsonl` contained exactly the four documents whose `stop_reasons` was `max_gen_toks` — the column and the files agree.

## Details

- The counts are gathered with `lm.gather_object`, like the sample metrics, so multi-rank runs report the same numbers as single-rank ones; the aggregation follows the same leaf-task rule as the parent/child subtask tables.
- Reviewers should scrutinise the disjointness rule and the `answer_found is False` clause: a backend that does not attach that attribute still gets empty/`None` detection, so the column degrades gracefully instead of being wrong.
- The extra columns change the printed table for every run. That is deliberate — the diagnostics are the point — and it is why the `test_printed_results` reference output has to move with it.
