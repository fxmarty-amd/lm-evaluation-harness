<!-- markdownlint-disable -->

## Disclosure

AI assistance was used. The changes were reviewed and tested manually.

## Overview

`repeats` exists in task configs, but the harness reports a single number for it: the samples of every repeat are pooled, so the run-to-run variance of a greedy or sampled run is invisible and the per-repeat samples are not written anywhere. Comparing two checkpoints then means comparing two pooled means with no idea whether the difference is larger than the noise.

This adds opt-in repeat reporting — an override for the number of repeats, per-repeat results and samples, min/mean/median/max across repeats — and, because a repeat list is exactly what a vote needs, a `maj@k` metric for the AIME tasks.

## Claims

- `--repeats N` (and `repeats:` in the evaluate config) overrides the task's own repeat count, and per-repeat output is implied by it: with one response per document there is nothing to report, with more than one there is no reason to ask separately, so no extra flag exists. Tasks that do not opt in (`repeats: 1`, the default) behave exactly as before.
- When `repeats > 1`, each repeat is written under its own directory, using the ordinary layout, next to the combined run:

```
<output_path>/<task>_<ts>/model__name/samples_<task>_<ts>.jsonl        # pooled
<output_path>/<ts>_repeat0/model__name/{results,samples}_<task>_<ts>.jsonl
<output_path>/<ts>_repeat1/model__name/{results,samples}_<task>_<ts>.jsonl
...
```

- When `repeats > 1`, every metric gets `_min_repeats`, `_mean_repeats`, `_median_repeats` and `_max_repeats` next to the pooled value, and the CLI prints them at DEBUG:

```
| Tasks  | Version | Filter | n-shot | Metric                       | Value  | Stderr |
|:-------|--------:|:-------|-------:|:-----------------------------|-------:|-------:|
| aime26 |       0 | none   |      0 | exact_match                  | 0.9333 | 0.0463 |
|        |         |        |        | exact_match_max_repeats      | 0.9333 |        |
|        |         |        |        | exact_match_mean_repeats     | 0.9000 |        |
|        |         |        |        | exact_match_median_repeats   | 0.9000 |        |
|        |         |        |        | exact_match_min_repeats      | 0.8333 |        |
```

- `repeat_results` and `repeat_samples` are added to the results document, so the per-repeat scores and samples are available to downstream tooling.
- Filters that need more than one response per document (`take_first_k` with k > 1) are skipped for the per-repeat statistics with a warning, instead of silently corrupting them.
- AIME tasks report `maj@k`: the majority-voted answer over the repeats, voted on normalized answers so equivalent forms of the same value count as one vote. `exact_match` becomes the share of correct samples (avg@k), which aggregated with `mean` is the standard "share of samples correct".

Left to other PRs:
- [ ] No `pass@k` / `pass@1` aggregations: `pass@1` would be the mean that already exists, and the oracle `pass@k` is not what a vote measures.
- [ ] `maj@k` is added to the AIME tasks present in this tree; newer AIME variants get it when they land.

## Validation

```
$ python -m pytest tests/test_evaluator.py -q -k repeat
2 passed, 8 deselected in 0.17s
$ python -m pytest tests/test_cli_subcommands.py -q
1 passed, 86 deselected in 0.02s
```

End-to-end, on 8× MI350 (ROCm 7.0): 30 reasoning problems, 5 repeats through an OpenAI-compatible endpoint, greedy, 100k-token budget. Per-repeat accuracy and the spread the pooled number hides:

| repeat | accuracy | problems at the token cap |
| --- | --- | --- |
| 0 | 0.933 | 1 |
| 1 | 0.900 | 3 |
| 2 | 0.933 | 1 |
| 3 | 0.900 | 3 |
| 4 | 0.833 | 5 |

which the harness reports as `exact_match = 0.9333` (pooled), `min 0.8333 / mean 0.9000 / median 0.9000 / max 0.9333`, i.e. the same task looked 10 points better or worse depending on which repeat you happened to run.

Reproduce `aime25` with 16 repeats against a served model:

```bash
# 1. serve
vllm serve Qwen/Qwen3.8-27B --served-model-name Qwen/Qwen3.8-27B \
    --host 127.0.0.1 --port 8100 \
    --tensor-parallel-size 8 --dtype bfloat16 \
    --max-model-len 65536 --gpu-memory-utilization 0.85 \
    --enable-prefix-caching > /tmp/qwen38_serve.log 2>&1 &

# 2. 16 samples per problem: maj@16 needs sampling, a greedy run would repeat itself
export MODEL_DIR=/path/to/Qwen/Qwen3.8-27B
lm_eval --model local-chat-completions \
    --model_args "model=Qwen/Qwen3.8-27B,base_url=http://127.0.0.1:8100/v1/chat/completions,num_concurrent=16,timeout=99999,tokenized_requests=False,think_end_token=,tokenizer=$MODEL_DIR,tokenizer_backend=huggingface" \
    --tasks aime25 --apply_chat_template \
    --gen_kwargs "max_gen_toks=32768,temperature=0.7,top_p=0.95" \
    --repeats 16 \
    --log_samples --output_path results/aime25_repeats
```

That prints `exact_match` (share of correct samples, i.e. avg@k) with the `_min_repeats`, `_mean_repeats`, `_median_repeats` and `_max_repeats` variants, plus `maj@k`, and writes each repeat under `results/aime25_repeats/<task>_<timestamp>_repeat<N>/` next to the pooled run.

`maj@k` behaviour, by calling the AIME `process_results` with one response per repeat (target 42):

| responses | exact_match (avg@k) | maj@k |
| --- | --- | --- |
| 5× `\boxed{42}` | 1.00 | 1 |
| 3× `\boxed{42}`, 2× `\boxed{7}` | 0.60 | 1 |
| 2× `\boxed{42}`, 3× `\boxed{7}` | 0.40 | 0 |
| `\boxed{42}`, `\boxed{42.0}`, 3× `\boxed{7}` | 0.40 | 1 (normalized vote) |
| single sample, correct / wrong | 1.00 / 0.00 | 1 / 0 |

## Details

- Each repeat is re-scored through the task's own `process_results`, so a task sees one response per repeat; `maj@k` therefore needs no new aggregation, since the per-document value is already 0/1 and `mean` aggregates it.
- The per-repeat bookkeeping dict is still named `repeat_diagnostic_stats`; it holds the per-repeat sample counts and predates this change.
- Reviewers should land this before the diagnostics PR: both extend `evaluate()`, `_cli/run.py` and `result_schema.py`, and this one restructures the result assembly they share.
- Rejected: computing `maj@k` in a post-processing script over the logged samples — it would be invisible to the CLI table and would have to re-implement the task's answer normalization.
